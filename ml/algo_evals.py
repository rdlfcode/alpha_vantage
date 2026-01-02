"""
AlgoEvals: Unified class for ML model training, optimization, and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import duckdb
import copy
import uuid
import optuna
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import timedelta
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from build_dataset import build_dataset
from ml.models import create_model
from ml.dataset import AlgoEvalsDataset
from ml.checkpoint_manager import CheckpointManager
from data.settings import settings as data_settings
from data.alpha_vantage_schema import TABLE_SCHEMAS


class AlgoEvals:
    """
    Unified class for managing model training, hyperparameter optimization,
    and prediction generation for algorithmic trading evaluations.
    """
    
    def __init__(self, settings: Dict[str, Any], db_conn: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize AlgoEvals with configuration settings.
        
        Args:
            settings: Configuration dictionary (typically ml_settings)
            db_conn: Optional database connection. If None, will create one.
        """
        self.settings = settings
        self.device = torch.device(settings['system']['device'])
        
        # Database connection
        self.local_conn = False
        if db_conn is None:
            db_path = Path(data_settings.get("data_dir"), data_settings.get("db_name"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_conn = duckdb.connect(str(db_path))
            self.local_conn = True
        else:
            self.db_conn = db_conn
            
        # Ensure required tables exist
        self._ensure_tables()
        
        # Data holders
        self.df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def _ensure_tables(self):
        """Ensure required database tables exist."""
        for table_name in ["PREDICTIONS", "MODEL_METADATA"]:
            if table_name in TABLE_SCHEMAS:
                create_sql = TABLE_SCHEMAS[table_name].replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                self.db_conn.execute(create_sql)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare the dataset for training.
        
        Returns:
            DataFrame containing the complete dataset
        """
        print("Loading dataset...")
        self.df = build_dataset()
        
        # Filter to specific features if requested
        feature_cols = self.settings['data'].get('feature_columns', ['all'])
        if 'all' not in feature_cols:
            cols_to_keep = list(set(
                feature_cols + 
                [self.settings['data']['target_col'], 
                 self.settings['data']['time_col'], 
                 self.settings['data']['group_col']]
            ))
            self.df = self.df[cols_to_keep]
        
        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        return self.df
    
    def prepare_datasets(self, df: Optional[pd.DataFrame] = None) -> Tuple[AlgoEvalsDataset, AlgoEvalsDataset, AlgoEvalsDataset]:
        """
        Split data and create train/val/test datasets.
        
        Args:
            df: DataFrame to split. If None, uses self.df
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.df
        
        time_col = self.settings['data']['time_col']
        # Sort by time
        df = df.sort_values(by=[time_col]).reset_index(drop=True)
        
        # Split data using unique timestamps to avoid cutting off symbols mid-timestamp
        unique_times = df[time_col].unique() # Already sorted since df is sorted
        n_unique = len(unique_times)
        
        train_split = self.settings['data']['train_split']
        val_split = self.settings['data']['val_split']
        
        train_end_idx = int(n_unique * train_split)
        val_end_idx = int(n_unique * (train_split + val_split))
        
        train_cutoff = unique_times[train_end_idx]
        val_cutoff = unique_times[val_end_idx] if val_end_idx < n_unique else unique_times[-1]
        
        train_df = df[df[time_col] < train_cutoff].copy()
        val_df = df[(df[time_col] >= train_cutoff) & (df[time_col] < val_cutoff)].copy()
        test_df = df[df[time_col] >= val_cutoff].copy()
        
        print(f"Split data by {time_col}:")
        print(f"  Train: {train_df[time_col].min()} to {train_df[time_col].max()} ({len(train_df)} rows)")
        print(f"  Val:   {val_df[time_col].min()} to {val_df[time_col].max()} ({len(val_df)} rows)")
        print(f"  Test:  {test_df[time_col].min()} to {test_df[time_col].max()} ({len(test_df)} rows)")
        
        # Create datasets
        self.train_dataset = AlgoEvalsDataset(
            train_df,
            target_col=self.settings['data']['target_col'],
            sequence_length=self.settings['data']['sequence_length'],
            prediction_horizon=self.settings['data']['prediction_horizon_training'],
            prediction_mode=self.settings['data']['prediction_mode'],
            scaler_type=self.settings['data']['scaler_type'],
            scaler_overrides=self.settings['data'].get('scaler_overrides', {})
        )
        
        self.val_dataset = AlgoEvalsDataset(
            val_df,
            target_col=self.settings['data']['target_col'],
            sequence_length=self.settings['data']['sequence_length'],
            prediction_horizon=self.settings['data']['prediction_horizon_training'],
            prediction_mode=self.settings['data']['prediction_mode'],
            scalers=self.train_dataset.scalers,
            encoders=self.train_dataset.encoders,
            scaler_type=self.settings['data']['scaler_type']
        )
        
        self.test_dataset = AlgoEvalsDataset(
            test_df,
            target_col=self.settings['data']['target_col'],
            sequence_length=self.settings['data']['sequence_length'],
            prediction_horizon=self.settings['data']['prediction_horizon_training'],
            prediction_mode=self.settings['data']['prediction_mode'],
            scalers=self.train_dataset.scalers,
            encoders=self.train_dataset.encoders,
            scaler_type=self.settings['data']['scaler_type']
        )
        
        # Create data loaders
        batch_size = self.settings['data']['batch_size']
        num_workers = self.settings['data']['num_workers']
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def calculate_metrics(self, y_true, y_pred, n_features=None):
        """
        Calculate evaluation metrics for predictions.
        
        For seq2seq mode, metrics are averaged across the sequence dimension if configured.
        
        Args:
            y_true: True values (tensor or numpy array)
            y_pred: Predicted values (tensor or numpy array)
            n_features: Number of features for adjusted R2 calculation
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert to numpy
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Handle seq2seq averaging
        prediction_mode = self.settings['data']['prediction_mode']
        average_sequence = self.settings['evaluation']['average_sequence_metrics']
        
        if prediction_mode == 'seq2seq' and average_sequence and y_true.ndim > 1:
            # Calculate metrics per time step, then average
            metrics_per_step = []
            for t in range(y_true.shape[-1]):
                y_t = y_true[..., t].flatten()
                y_p = y_pred[..., t].flatten()
                metrics_per_step.append(self._compute_metrics(y_t, y_p, n_features))
            
            # Average across time steps
            metrics = {}
            for key in metrics_per_step[0].keys():
                values = [m[key] for m in metrics_per_step if not np.isnan(m[key])]
                metrics[key] = np.mean(values) if values else np.nan
                
            return metrics
        else:
            # Flatten and compute metrics
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            return self._compute_metrics(y_true, y_pred, n_features)
    
    def _compute_metrics(self, y_true, y_pred, n_features=None):
        """Compute individual metrics on flat arrays."""
        metrics = {}
        
        # MAE
        metrics["MAE"] = np.mean(np.abs(y_true - y_pred))
        
        # MSE
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["MSE"] = mse
        
        # RMSE
        metrics["RMSE"] = np.sqrt(mse)
        
        # MAPE
        mask = y_true != 0
        if mask.sum() == 0:
            metrics["MAPE"] = 0.0
        else:
            metrics["MAPE"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        metrics["R2"] = r2
        
        # Adjusted R-squared
        if n_features is not None:
            n = len(y_true)
            p = n_features
            if n > p + 1:
                metrics["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            else:
                metrics["Adj_R2"] = np.nan
        else:
            metrics["Adj_R2"] = np.nan
        
        # RMSLE
        y_true_clipped = np.maximum(y_true, 0)
        y_pred_clipped = np.maximum(y_pred, 0)
        metrics["RMSLE"] = np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))
        
        return metrics
    
    def train_one_epoch(self, model, loader, optimizer, criterion):
        """Train model for one epoch."""
        model.train()
        total_loss = 0
        
        for batch in loader:
            x_num = batch['x_num'].to(self.device)
            y = batch['y'].to(self.device)
            
            optimizer.zero_grad()
            output = model(x_num)
            
            # Align shapes for loss calculation
            if output.shape != y.shape:
                if output.shape[-1] == 1 and y.ndim == 1:
                    y = y.unsqueeze(-1)
                elif y.shape[-1] == 1 and output.ndim > y.ndim:
                    y = y.squeeze(-1)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, model, loader, criterion):
        """Evaluate model on a dataset."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        n_features = None
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x_num = batch['x_num'].to(self.device)
                y = batch['y'].to(self.device)
                
                # Capture feature dimension from first batch
                if i == 0:
                    if x_num.ndim == 3:
                        n_features = x_num.shape[1] * x_num.shape[2]
                    elif x_num.ndim == 2:
                        n_features = x_num.shape[1]
                
                output = model(x_num)
                
                # Align shapes
                if output.shape != y.shape:
                    if output.shape[-1] == 1 and y.ndim == 1:
                        y = y.unsqueeze(-1)
                    elif y.shape[-1] == 1 and output.ndim > y.ndim:
                        y = y.squeeze(-1)
                
                loss = criterion(output, y)
                total_loss += loss.item()
                
                all_preds.append(output)
                all_targets.append(y)
        
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        
        if len(all_preds) > 0:
            y_pred = torch.cat(all_preds)
            y_true = torch.cat(all_targets)
            metrics = self.calculate_metrics(y_true, y_pred, n_features=n_features)
            metrics['Loss'] = avg_loss
        else:
            metrics = {"Loss": avg_loss}
        
        return metrics
    
    def train_single_model(
        self, 
        model_type: Optional[str] = None, 
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train a single model with specified or default hyperparameters.
        
        Args:
            model_type: Type of model to train. If None, uses settings['model']['type']
            hyperparams: Optional hyperparameters to override defaults
            
        Returns:
            Dictionary containing training history and final metrics
        """
        if self.train_loader is None:
            raise ValueError("Datasets not prepared. Call prepare_datasets() first.")
        
        # Setup model configuration
        model_settings = copy.deepcopy(self.settings)
        if model_type:
            model_settings['model']['type'] = model_type
        
        if hyperparams:
            # Apply hyperparameters
            for key, value in hyperparams.items():
                if key in ['learning_rate', 'batch_size', 'epochs']:
                    model_settings['training'][key] = value
                else:
                    # Model-specific parameter
                    mt = model_settings['model']['type']
                    if mt not in model_settings['model']['params']:
                        model_settings['model']['params'][mt] = {}
                    model_settings['model']['params'][mt][key] = value
        
        # Set input dimension
        num_vals, _ = self.train_dataset.get_feature_dims()
        model_settings['model']['input_dim'] = num_vals
        
        # Adjust output dimension for seq2seq
        if model_settings['data']['prediction_mode'] == 'seq2seq':
            horizon = model_settings['data']['prediction_horizon_training']
            model_settings['model']['output_dim'] = horizon
        
        # Create model
        print(f"Initializing Model: {model_settings['model']['type']} with input_dim={num_vals}")
        model = create_model(model_settings)
        model.to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        lr = model_settings['training'].get('learning_rate', 1e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        epochs = model_settings['training'].get('epochs', 100)
        patience = model_settings['training'].get('early_stopping_patience', 10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        
        print(f"Starting training on {self.device} with LR={lr}...")
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            model_name=model_settings.get('model', {}).get('name', model_settings['model']['type']),
            model_type=model_settings['model']['type'],
            checkpoint_dir=model_settings['training'].get('checkpoint_dir', 'ml/checkpoints'),
            settings=model_settings,
            db_conn=self.db_conn
        )
        
        dataset_size = len(self.train_loader.dataset)
        
        pbar = tqdm(range(epochs), desc=f"Training {model_settings['model']['type']}", leave=False)
        for epoch in pbar:
            train_loss = self.train_one_epoch(model, self.train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, self.val_loader, criterion)
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                **val_metrics
            })
            
            pbar.set_postfix({
                'T_Loss': f"{train_loss:.4f}",
                'V_Loss': f"{val_metrics['Loss']:.4f}",
                'MAPE': f"{val_metrics.get('MAPE', 0):.2f}%"
            })
            
            # Early stopping and checkpointing
            if val_metrics['Loss'] < best_val_loss:
                best_val_loss = val_metrics['Loss']
                patience_counter = 0
                checkpoint_manager.save(
                    model=model,
                    metrics=val_metrics,
                    dataset_size=dataset_size,
                    epoch=epoch
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    pbar.write("Early stopping triggered")
                    break
        
        return {
            'model': model,
            'history': history,
            'best_val_loss': best_val_loss,
            'model_type': model_settings['model']['type']
        }
    
    def optimize_model(self, model_type: str, n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model type using Optuna.
        
        Args:
            model_type: Type of model to optimize
            n_trials: Number of optimization trials. If None, uses settings value
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        if model_type not in self.settings.get('hyperparameters', {}):
            print(f"No hyperparameter configuration found for model type: {model_type}")
            return {}
        
        if n_trials is None:
            n_trials = self.settings['optimization']['n_trials']
        
        hp_config = self.settings['hyperparameters'][model_type]
        
        print(f"\nStarting Optuna Optimization for {model_type} with {n_trials} trials...")
        
        def suggest_parameter(trial, name, value_config):
            """Helper to suggest parameters based on config type."""
            if isinstance(value_config, list):
                return trial.suggest_categorical(name, value_config)
            
            if isinstance(value_config, tuple) and len(value_config) == 2:
                start, end = value_config
                if isinstance(start, int) and isinstance(end, int):
                    return trial.suggest_int(name, start, end)
                else:
                    # Use log scale for wide ranges
                    log = False
                    if start > 0 and end > 0 and (end / start) >= 100:
                        log = True
                    return trial.suggest_float(name, start, end, log=log)
            
            return value_config
        
        def objective(trial):
            # Build hyperparameters for this trial
            trial_hyperparams = {}
            for key, config in hp_config.items():
                trial_hyperparams[key] = suggest_parameter(trial, key, config)
            
            # Train model with these hyperparameters
            try:
                result = self.train_single_model(model_type=model_type, hyperparams=trial_hyperparams)
            except Exception as e:
                print(f"Trial failed: {e}")
                raise optuna.TrialPruned()
            
            # Return metric to optimize
            optimization_metric = self.settings['optimization']['optimization_metric']
            
            if not result['history']:
                raise optuna.TrialPruned()
            
            # Get best value of the optimization metric
            metric_values = [h.get(optimization_metric, float('inf')) for h in result['history']]
            best_metric = min(metric_values) if self.settings['optimization']['optimization_direction'] == 'minimize' else max(metric_values)
            
            return best_metric
        
        # Create and run study
        direction = self.settings['optimization']['optimization_direction']
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("\n" + "="*60)
        print(f"Optimization Complete for {model_type}")
        print(f"Best {self.settings['optimization']['optimization_metric']}: {study.best_value:.4f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def train_all_models(self, optimize: bool = True) -> List[Dict[str, Any]]:
        """
        Train all configured models, with optional hyperparameter optimization.
        
        Args:
            optimize: If True, optimize hyperparameters before final training
            
        Returns:
            List of results for each model trained
        """
        models_to_train = self.settings['optimization']['models_to_optimize']
        
        if 'all' in models_to_train:
            models_to_train = list(self.settings['hyperparameters'].keys())
        
        results = []
        
        for model_type in models_to_train:
            print(f"\n{'='*60}")
            print(f"Processing Model: {model_type}")
            print(f"{'='*60}\n")
            
            if optimize:
                # Optimize hyperparameters
                opt_result = self.optimize_model(model_type)
                best_params = opt_result.get('best_params', {})
                
                # Train final model with best parameters
                print(f"\nTraining final {model_type} model with optimized parameters...")
                final_result = self.train_single_model(model_type=model_type, hyperparams=best_params)
                final_result['optimization'] = opt_result
            else:
                # Train with default parameters
                final_result = self.train_single_model(model_type=model_type)
            
            results.append(final_result)
        
        return results
    
    def generate_predictions(
        self, 
        model, 
        horizon_days: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate future predictions using a trained model.
        
        Args:
            model: Trained model to use for predictions
            horizon_days: Number of days to predict. If None, uses settings value
            model_name: Name for the model in predictions table
            
        Returns:
            DataFrame with predictions
        """
        if horizon_days is None:
            horizon_days = self.settings['data']['prediction_horizon_inference']
        
        if model_name is None:
            model_name = self.settings['model']['type']
        
        print(f"\nGenerating {horizon_days}-day predictions...")
        
        model.eval()
        model.to(self.device)
        
        seq_len = self.settings['data']['sequence_length']
        target_col = self.settings['data']['target_col']
        group_col = self.settings['data']['group_col']
        time_col = self.settings['data']['time_col']
        
        # Get last window for each symbol
        last_windows = self.df.groupby(group_col).tail(seq_len)
        
        # Filter symbols with sufficient history
        symbol_counts = last_windows.groupby(group_col).size()
        valid_symbols = symbol_counts[symbol_counts == seq_len].index
        last_windows = last_windows[last_windows[group_col].isin(valid_symbols)]
        
        if last_windows.empty:
            print("No symbols have enough history for prediction.")
            return pd.DataFrame()
        
        # Create dataset for inference
        pred_dataset = AlgoEvalsDataset(
            last_windows,
            target_col=target_col,
            sequence_length=seq_len,
            prediction_horizon=1,
            scalers=self.train_dataset.scalers,
            encoders=self.train_dataset.encoders,
            is_inference=True,
            scaler_type=self.settings['data']['scaler_type']
        )
        
        # Get target column index
        try:
            target_idx = pred_dataset.num_cols.index(target_col)
        except ValueError:
            raise ValueError(f"Target column {target_col} not found in numerical columns.")
        
        # Prepare input batch
        num_features = len(pred_dataset.num_cols)
        X_batch = torch.tensor(
            pred_dataset.processed_data[pred_dataset.num_cols].values,
            dtype=torch.float32
        ).view(-1, seq_len, num_features).to(self.device)
        
        unique_symbols = pred_dataset.processed_data[group_col].unique()
        last_dates = last_windows.groupby(group_col)[time_col].max()
        last_dates = last_dates.loc[unique_symbols].to_dict()
        
        all_predictions = []
        run_id = str(uuid.uuid4())
        now_ts = pd.Timestamp.now()
        target_scaler = self.train_dataset.scalers[target_col]
        
        # Generate predictions iteratively
        for i in range(1, horizon_days + 1):
            with torch.no_grad():
                preds_scaled = model(X_batch)
                
                # Handle different output shapes
                if preds_scaled.ndim == 3:
                    # seq2seq output: (batch, seq, features) - take last time step
                    preds_scaled = preds_scaled[:, -1, :]
                
                if preds_scaled.shape[-1] > 1:
                    # Multi-output, take first (assuming it's the target)
                    preds_scaled = preds_scaled[:, 0:1]
            
            # Inverse transform
            preds_np = preds_scaled.cpu().numpy()
            preds_real = target_scaler.inverse_transform(preds_np)
            
            # Store predictions
            for idx, symbol in enumerate(unique_symbols):
                next_date = last_dates[symbol] + timedelta(days=i)
                all_predictions.append({
                    "symbol": symbol,
                    "dt": next_date,
                    "predicted_value": float(preds_real[idx][0]),
                    "model_name": model_name,
                    "run_id": run_id,
                    "created_at": now_ts
                })
            
            # Update input for next step
            last_step = X_batch[:, -1, :].clone()
            
            if preds_scaled.ndim == 1:
                preds_scaled = preds_scaled.unsqueeze(1)
            
            last_step[:, target_idx] = preds_scaled[:, 0]
            next_step_unsqueezed = last_step.unsqueeze(1)
            X_batch = torch.cat([X_batch[:, 1:, :], next_step_unsqueezed], dim=1)
        
        print(f"Generated {len(all_predictions)} predictions.")
        
        # Save to database
        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            self.db_conn.register("pred_df_temp", pred_df)
            self.db_conn.execute("INSERT OR REPLACE INTO PREDICTIONS SELECT * FROM pred_df_temp")
            self.db_conn.unregister("pred_df_temp")
            print("Predictions saved to database.")
            
            return pred_df
        
        return pd.DataFrame()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.local_conn and self.db_conn:
            self.db_conn.close()
        # If an exception occurred, you could handle it here.
        if exc_type:
            print(f"An exception occurred: {exc_val}")
            return False # Re-raise the exception after cleanup
        return True # Suppress no exception

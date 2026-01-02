"""
Training and Evaluation loops with support for Generic Settings and Fine-tuning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import duckdb
from pathlib import Path
from ml.checkpoint_manager import CheckpointManager
import sys
import uuid
import json
import pandas as pd
from datetime import timedelta
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_dataset import build_dataset
from ml.models import create_model
from ml.dataset import AlgoEvalsDataset
from ml.settings import ml_settings
from data.settings import settings as data_settings
from data.alpha_vantage_schema import TABLE_SCHEMAS

def calculate_metrics(y_true, y_pred, n_features=None):
    # Move to cpu numpy
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Ensure they are flat or aligned
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    metrics = {}
    
    # 1. MAE
    metrics["MAE"] = np.mean(np.abs(y_true - y_pred))
    
    # 2. MSE
    mse = np.mean((y_true - y_pred) ** 2)
    metrics["MSE"] = mse
    
    # 3. RMSE
    metrics["RMSE"] = np.sqrt(mse)
    
    # 4. MAPE
    mask = y_true != 0
    if mask.sum() == 0:
        metrics["MAPE"] = 0.0
    else:
        metrics["MAPE"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    # 5. R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        r2 = 0.0 # Constant target
    else:
        r2 = 1 - (ss_res / ss_tot)
    metrics["R2"] = r2
    
    # 6. Adjusted R-squared
    if n_features is not None:
        n = len(y_true)
        p = n_features
        if n > p + 1:
            metrics["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            metrics["Adj_R2"] = np.nan # Not enough data points
    else:
        metrics["Adj_R2"] = np.nan
        
    # 7. RMSLE
    # Log requires non-negative. Clip to 0.
    y_true_clipped = np.maximum(y_true, 0)
    y_pred_clipped = np.maximum(y_pred, 0)
    metrics["RMSLE"] = np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))
    
    return metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        # batch is a dict
        x_num = batch['x_num'].to(device)
        # x_cat = batch['x_cat'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(x_num)
        
        # y shape alignment
        if output.shape[-1] == 1 and y.ndim == 1:
            y = y.unsqueeze(-1)
            
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # To calculate number of features for Adj R2
    # We assume x_num shape is (batch, seq, feat)
    n_features = None 
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_num = batch['x_num'].to(device)
            y = batch['y'].to(device)
            
            # Capture feature dim from first batch
            if i == 0:
                if x_num.ndim == 3:
                    n_features = x_num.shape[1] * x_num.shape[2]
                elif x_num.ndim == 2:
                    n_features = x_num.shape[1]
            
            output = model(x_num)
            
            if output.shape[-1] == 1 and y.ndim == 1:
                y = y.unsqueeze(-1)
                
            loss = criterion(output, y)
            total_loss += loss.item()
            
            all_preds.append(output)
            all_targets.append(y)
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    if len(all_preds) > 0:
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        metrics = calculate_metrics(y_true, y_pred, n_features=n_features)
        metrics['Loss'] = avg_loss
    else:
        metrics = {"Loss": avg_loss}
    
    return metrics

def train_model(model, train_loader, val_loader, settings: Dict[str, Any], db_conn: Optional[duckdb.DuckDBPyConnection] = None):
    """
    Generic training loop that uses settings dict.
    Supports fine-tuning via settings['training']['fine_tune'].
    """
    training_settings = settings['training']
    device = torch.device(settings['system']['device'])
    model.to(device)
    
    criterion = nn.MSELoss()
    
    # Fine-tuning logic
    lr = training_settings.get("learning_rate", 1e-3)
    if training_settings.get("fine_tune", False):
        print("Fine-tuning mode enabled.")
        lr = training_settings.get("fine_tune_learning_rate", lr / 10)
        
        if training_settings.get("freeze_encoder", False):
            print("Freezing encoder layers...")
            # Heuristic to freeze obvious encoder layers if they exist
            # For TITANS/Transformer:
            if hasattr(model, 'core_transformer'):
                for param in model.core_transformer.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'transformer_encoder'):
                 for param in model.transformer_encoder.parameters():
                    param.requires_grad = False
            # For Models with 'linear' feature extractor?
            # Adjust as necessary for specific architectures

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = training_settings.get("epochs", 100)
    patience = training_settings.get("early_stopping_patience", 10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    print(f"Starting training on {device} with LR={lr}...")
    
    # Initialize Checkpoint Manager
    # Ensure we have a DB connection if not provided
    local_conn = False
    if db_conn is None:
        db_path = Path(settings.get("data", {}).get("data_dir", "data")) / settings.get("db_name", "alpha_vantage.db")
        try:
             # Just in case, ensuring parent dir exists
             db_path.parent.mkdir(parents=True, exist_ok=True)
             db_conn = duckdb.connect(str(db_path))
             local_conn = True
        except Exception as e:
            print(f"Warning: Could not connect to DB for metadata logging: {e}")

    dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0

    checkpoint_manager = CheckpointManager(
        model_name=settings.get("model", {}).get("name", "model"),
        model_type=settings.get("model", {}).get("type", "unknown"),
        checkpoint_dir=settings.get("training", {}).get("checkpoint_dir", "ml/checkpoints"),
        settings=settings,
        db_conn=db_conn
    )
    
    pbar = tqdm(range(epochs), desc="Training", leave=False)
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        pbar.set_postfix({
            'T_Loss': f"{train_loss:.4f}",
            'V_Loss': f"{val_metrics['Loss']:.4f}",
            'MAPE': f"{val_metrics['MAPE']:.2f}%"
        })
        
        # Early stopping
        if val_metrics['Loss'] < best_val_loss:
            best_val_loss = val_metrics['Loss']
            patience_counter = 0
            # torch.save(model.state_dict(), "best_model.pth")
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
                
    if local_conn and db_conn:
        db_conn.close()

    return history

def train_and_predict(settings):
    print("Loading dataset...")
    df = build_dataset()
    
    feature_cols = settings['data'].get('feature_columns', ['all'])
    if 'all' not in feature_cols:
        cols_to_keep = list(set(feature_cols + [settings['data']['target_col'], settings['data']['time_col'], settings['data']['group_col']]))
        df = df[cols_to_keep]
        
    train_split = settings['data']['train_split']
    time_col = settings['data']['time_col']
    df = df.sort_values(by=[time_col]).reset_index(drop=True)
    
    unique_times = df[time_col].unique()
    n_unique = len(unique_times)
    train_end_idx = int(n_unique * train_split)
    train_cutoff = unique_times[train_end_idx]
    
    train_df = df[df[time_col] < train_cutoff].copy()
    val_df = df[df[time_col] >= train_cutoff].copy()
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    train_dataset = AlgoEvalsDataset(
        train_df, 
        target_col=settings['data']['target_col'],
        sequence_length=settings['data']['sequence_length'],
        prediction_horizon=settings['data']['prediction_horizon']
    )
    
    val_dataset = AlgoEvalsDataset(
        val_df, 
        target_col=settings['data']['target_col'],
        sequence_length=settings['data']['sequence_length'],
        prediction_horizon=settings['data']['prediction_horizon'],
        scalers=train_dataset.scalers, 
        encoders=train_dataset.encoders 
    )
    
    batch_size = settings['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=settings['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=settings['data']['num_workers'])
    
    num_vals, _ = train_dataset.get_feature_dims()
    settings['model']['input_dim'] = num_vals 
    
    print(f"Initializing Model: {settings['model']['type']} with input_dim={num_vals}")
    model = create_model(settings)
    
    db_path = Path(data_settings.get("data_dir"), data_settings.get("db_name"))
    db_conn = duckdb.connect(str(db_path))
    
    for table_name in ["PREDICTIONS", "MODEL_METADATA"]:
        if table_name in TABLE_SCHEMAS:
             create_sql = TABLE_SCHEMAS[table_name].replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
             db_conn.execute(create_sql)
    
    history = train_model(model, train_loader, val_loader, settings, db_conn=db_conn)
    
    print("Generating Future Predictions...")
    model.eval()
    device = torch.device(settings['system']['device'])
    model.to(device)
    
    # --- Batch Prediction Optimization ---
    horizon_days = settings['data']['prediction_horizon_days']
    seq_len = settings['data']['sequence_length']
    target_col = settings['data']['target_col']
    
    # 1. Prepare Last Window for ALL symbols
    # We create a dataframe containing the last seq_len rows for each symbol
    last_windows = df.groupby(settings['data']['group_col']).tail(seq_len)
    
    # Filter out symbols with insufficient history
    symbol_counts = last_windows.groupby(settings['data']['group_col']).size()
    valid_symbols = symbol_counts[symbol_counts == seq_len].index
    last_windows = last_windows[last_windows[settings['data']['group_col']].isin(valid_symbols)]
    
    if last_windows.empty:
        print("No symbols have enough history for prediction.")
        db_conn.close()
        return

    # 2. Create Dataset for these windows (handling scaling efficiently)
    # We use AlgoEvalsDataset to process features correctly, but we won't use its iterator for the loop
    pred_dataset = AlgoEvalsDataset(
        last_windows,
        target_col=target_col,
        sequence_length=seq_len,
        prediction_horizon=1, 
        scalers=train_dataset.scalers,
        encoders=train_dataset.encoders,
        is_inference=True
    )
    
    # Extract Tensor Batch: (Num_Symbols, Seq_Len, Features)
    # The dataset creates sequences. Since we have exactly seq_len per symbol, 
    # we expect exactly 1 sequence per symbol if we adjust logic, or we just extract raw processed data.
    # AlgoEvalsDataset flattens by symbol. 
    # Let's extract processed values and reshape.
    
    # Sort by symbol to align with our valid_symbols list
    # processed_data is already sorted by group_col in init
    
    # We need to map target column to index in num_vals
    try:
        target_idx = pred_dataset.num_cols.index(target_col)
    except ValueError:
        raise ValueError(f"Target column {target_col} not found in numerical columns.")

    # Reshape processed data into (N, Seq, Feat)
    # processed_data has shape (N * Seq, Feat)
    num_features = len(pred_dataset.num_cols)
    X_batch = torch.tensor(
        pred_dataset.processed_data[pred_dataset.num_cols].values, 
        dtype=torch.float32
    ).view(-1, seq_len, num_features).to(device) # (N_Sym, Seq, Feat)
    
    # Keep track of meta data
    # unique_symbols will match the order of X_batch because dataset sorts by group
    unique_symbols = pred_dataset.processed_data[settings['data']['group_col']].unique()
    
    # Track Last Date per symbol
    last_dates = last_windows.groupby(settings['data']['group_col'])[settings['data']['time_col']].max()
    last_dates = last_dates.loc[unique_symbols].to_dict() # Ensure order
    
    all_predictions = []
    run_id = str(uuid.uuid4())
    model_name = settings['model']['name'] if 'name' in settings['model'] else settings['model']['type']
    now_ts = pd.Timestamp.now()
    
    target_scaler = train_dataset.scalers[target_col]
    
    for i in tqdm(range(1, horizon_days + 1), desc="Predicting"):
        with torch.no_grad():
            # Shape: (N_Sym, 1) -- assuming output_dim=1
            preds_scaled = model(X_batch) 
            
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
            
        # Prepare Next Input
        # Shift: Drop first time step, Append new
        # Strategy: New step = Copy of Last Step, but with Target updated
        last_step = X_batch[:, -1, :].clone() # (N, Feat)
        
        # Update target feature with scaled prediction
        # Note: model output might be (N, 1) or (N)
        if preds_scaled.ndim == 1:
            preds_scaled = preds_scaled.unsqueeze(1)
            
        last_step[:, target_idx] = preds_scaled[:, 0]
        
        # Reshape to (N, 1, Feat) and Concat
        next_step_unsqueezed = last_step.unsqueeze(1)
        X_batch = torch.cat([X_batch[:, 1:, :], next_step_unsqueezed], dim=1)

    print(f"Generated {len(all_predictions)} predictions.")
    
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        db_conn.register("pred_df_temp", pred_df)
        db_conn.execute(f"INSERT OR REPLACE INTO PREDICTIONS SELECT * FROM pred_df_temp")
        db_conn.unregister("pred_df_temp")
        print("Predictions saved to DB.")
        
    db_conn.close()

if __name__ == "__main__":
    train_and_predict(ml_settings)

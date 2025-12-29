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
        x_cat = batch['x_cat'].to(device)
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
    device_name = settings['system']['device']
    
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
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
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['Loss']:.4f} - MAPE: {val_metrics['MAPE']:.2f}%")
        
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
                print("Early stopping triggered")
                break
                
    if local_conn and db_conn:
        db_conn.close()

    return history

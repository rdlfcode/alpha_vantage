"""
Training and Evaluation loops with support for Generic Settings and Fine-tuning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

def calculate_metrics(y_true, y_pred):
    # Move to cpu numpy
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    # Avoid zero division
    mask = y_true != 0
    if mask.sum() == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    mse = np.mean((y_true - y_pred) ** 2)
    return {"MAPE": mape, "MSE": mse}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
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
    
    with torch.no_grad():
        for batch in loader:
            x_num = batch['x_num'].to(device)
            y = batch['y'].to(device)
            
            output = model(x_num)
            
            if output.shape[-1] == 1 and y.ndim == 1:
                y = y.unsqueeze(-1)
                
            loss = criterion(output, y)
            total_loss += loss.item()
            
            all_preds.append(output)
            all_targets.append(y)
            
    avg_loss = total_loss / len(loader)
    
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    metrics = calculate_metrics(y_true, y_pred)
    metrics['Loss'] = avg_loss
    
    return metrics

def train_model(model, train_loader, val_loader, settings: Dict[str, Any]):
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
            # Save best model state?
            # torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
    return history

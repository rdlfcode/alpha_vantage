# AlgoEvals Settings Reference

Quick reference for all configurable settings in `ml/settings.py`.

## System Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `device` | str | auto-detect | Computing device: "cpu", "cuda", or "mps" |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Data Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `target_col` | str | "close" | Target variable column name |
| `time_col` | str | "dt" | Timestamp column name |
| `group_col` | str | "symbol" | Entity grouping column (e.g., stock symbol) |
| `sequence_length` | int | 60 | Historical time steps for input |
| `prediction_mode` | str | "seq2seq" | "seq2seq" or "seq2point" |
| `prediction_horizon_training` | int | 5 | Output sequence length during training |
| `prediction_horizon_inference` | int | 91 | Days to predict in production |
| `train_split` | float | 0.8 | Fraction of data for training |
| `val_split` | float | 0.1 | Fraction of remaining data for validation |
| `batch_size` | int | 32 | Samples per training batch |
| `num_workers` | int | 0 | Data loader worker processes |
| `categorical_cols` | list | [] | Categorical column names (auto-detect if empty) |
| `numerical_cols` | list | [] | Numerical column names (auto-detect if empty) |
| `feature_columns` | list | ["all"] | Features to use (["all"] or specific list) |
| `scaler_type` | str | "standard" | Scaler type: "standard", "minmax", "robust", "none" |
| `scaler_overrides` | dict | {} | Per-column scaler overrides |

## Model Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | str | "TITANS" | Model architecture |
| `input_dim` | int | auto | Input feature dimension (set automatically) |
| `output_dim` | int | 1 | Output dimension (1 for seq2point, N for seq2seq) |

### Model-Specific Parameters

#### Transformer
- `d_model`: 128 - Embedding dimension
- `nhead`: 8 - Number of attention heads
- `num_layers`: 2 - Number of encoder layers
- `dropout`: 0.1 - Dropout rate

#### TITANS
- `d_model`: 128 - Embedding dimension
- `nhead`: 8 - Number of attention heads
- `num_layers`: 2 - Number of encoder layers
- `dropout`: 0.1 - Dropout rate
- `memory_type`: "neural" - Memory mechanism ("neural", "lstm", "transformer")
- `memory_size`: 256 - Long-term memory size

## Hyperparameter Ranges

Ranges for optimization (Optuna):

### TITANS
- `d_model`: [64, 128, 256]
- `nhead`: [4, 8]
- `num_layers`: (2, 6) - continuous range
- `dropout`: (0.1, 0.3)
- `memory_size`: [128, 256, 512]
- `learning_rate`: (1e-5, 1e-3) - log scale

### Transformer
- `d_model`: [64, 128, 256]
- `nhead`: [4, 8]
- `num_layers`: (2, 6)
- `dropout`: (0.1, 0.3)
- `learning_rate`: (1e-5, 1e-3)

## Training Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `epochs` | int | 100 | Maximum training epochs |
| `learning_rate` | float | 1e-4 | Optimizer learning rate |
| `early_stopping_patience` | int | 10 | Epochs to wait before early stopping |
| `fine_tune` | bool | False | Enable fine-tuning mode |
| `fine_tune_learning_rate` | float | 1e-5 | Learning rate for fine-tuning |
| `freeze_encoder` | bool | False | Freeze encoder during fine-tuning |
| `max_checkpoints_per_model` | int | 1 | Number of checkpoints to keep |
| `checkpoint_dir` | str | "ml/checkpoints" | Directory for model checkpoints |

## Optimization Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `n_trials` | int | 20 | Optuna trials per model |
| `models_to_optimize` | list | ["TITANS"] | Models to optimize (["all"] for all) |
| `optimization_metric` | str | "Loss" | Metric to optimize |
| `optimization_direction` | str | "minimize" | "minimize" or "maximize" |

## Evaluation Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `average_sequence_metrics` | bool | True | Average metrics across time steps in seq2seq mode |

## Common Configurations

### High-Performance Setup
```python
ml_settings['data']['batch_size'] = 64
ml_settings['training']['epochs'] = 200
ml_settings['optimization']['n_trials'] = 50
ml_settings['model']['params']['TITANS']['d_model'] = 256
```

### Fast Prototyping
```python
ml_settings['training']['epochs'] = 20
ml_settings['optimization']['n_trials'] = 5
ml_settings['data']['batch_size'] = 16
```

### Seq2point Mode
```python
ml_settings['data']['prediction_mode'] = 'seq2point'
ml_settings['data']['prediction_horizon_training'] = 1
ml_settings['model']['output_dim'] = 1
```

### Custom Scaling Strategy
```python
ml_settings['data']['scaler_type'] = 'robust'  # Default to RobustScaler
ml_settings['data']['scaler_overrides'] = {
    'close': 'standard',      # StandardScaler for close prices
    'volume': 'minmax',       # MinMaxScaler for volume
    'sentiment': 'none',      # No scaling for sentiment scores
}
```

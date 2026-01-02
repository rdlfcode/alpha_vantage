import torch


ml_settings = {
    "system": {
        # Configured for: "cpu", "cuda", or "mps". Auto-detected below.
        "device": None,
        # Random seed for reproducibility across runs
        "random_seed": 42,
    },
    
    "data": {
        # Column name containing the target variable to predict
        "target_col": "close",
        # Column name containing timestamps for temporal ordering
        "time_col": "dt",
        # Column name for grouping data by entity (e.g., stock symbol)
        "group_col": "symbol",
        # Number of historical time steps to use as input for predictions
        # Example: 60 means use 60 days of history to predict future values
        "sequence_length": 252,
        # Prediction mode: "seq2seq" for sequence-to-sequence, "seq2point" for sequence-to-point
        # - seq2seq: Model outputs a sequence of future values (e.g., next N days)
        # - seq2point: Model outputs a single future value
        "prediction_mode": "seq2seq",
        # Number of time steps ahead to predict during training
        # For seq2seq: this is the length of the output sequence during training
        # For seq2point: this is how many steps ahead to predict (typically 1)
        "prediction_horizon_training": 5,
        # Number of days to predict into the future during inference (after training)
        # This is used when generating predictions on new data
        "prediction_horizon_inference": 91,
        # Fraction of data to use for training (0.0 to 1.0)
        "train_split": 0.7,
        # Fraction of remaining data to use for validation (0.0 to 1.0)
        # Test split is automatically the remainder: 1.0 - train_split - val_split
        "val_split": 0.15,
        # Number of samples per batch during training
        "batch_size": 32,
        # Number of worker processes for data loading (0 = main process only)
        "num_workers": 0,
        # List of categorical column names, or empty list for auto-detection
        "categorical_cols": [],
        # List of numerical column names, or empty list for auto-detection
        "numerical_cols": [],
        # Feature columns to use: ["all"] to use all available, or specify list of column names
        "feature_columns": ["all"],
        # Type of scaler to use for numerical features
        # Options: "standard" (StandardScaler), "minmax" (MinMaxScaler), 
        #          "robust" (RobustScaler), "none" (no scaling)
        "scaler_type": "standard",
        # Optional: Dictionary to override scaler type for specific columns
        # Example: {"volume": "minmax", "price": "standard"}
        "scaler_overrides": {},
    },
    
    "model": {
        # Model architecture to use
        # Options: "Linear", "Exponential", "FFT", "Transformer", "TITANS"
        "type": "TITANS",
        # Input dimension - automatically set based on dataset features
        "input_dim": None,
        # Output dimension: 1 for single-value prediction point
        # For seq2seq mode, this gets multiplied by prediction_horizon_training
        "output_dim": 1,
        
        # Model-specific hyperparameters organized by model type
        "params": {
            # Linear baseline model - no additional parameters needed
            "Linear": {},
            # Exponential smoothing model - no additional parameters needed
            "Exponential": {},
            # 1D FFT (Fast Fourier Transform) model - operates on target sequence only
            "FFT1D": {},
            # 2D FFT model - operates on all features
            "FFT2D": {},
            # Transformer model parameters
            "Transformer": {
                # Dimensionality of transformer model embeddings
                "d_model": 128,
                # Number of attention heads in multi-head attention
                "nhead": 8,
                # Number of transformer encoder layers
                "num_layers": 2,
                # Dropout probability for regularization
                "dropout": 0.1,
            },
            
            # TITANS (Time-series Transformer with Adaptive Neural Storage) model parameters
            "TITANS": {
                # Dimensionality of the model embeddings
                "d_model": 128,
                # Number of attention heads
                "nhead": 8,
                # Number of encoder layers (short-term memory)
                "num_layers": 2,
                # Dropout rate for regularization
                "dropout": 0.1,
                # Type of long-term memory mechanism
                # Options: "neural" (feedforward), "lstm", "transformer"
                "memory_type": "neural",
                # Size of the long-term memory buffer
                "memory_size": 256,
            }
        }
    },
    
    # Hyperparameter tuning ranges for optimization
    # Only models listed here will be tuned during optimization
    # Format: List for categorical choices, Tuple for continuous ranges
    "hyperparameters": {
        "TITANS": {
            # Categorical: choose from discrete values
            "d_model": [64, 128, 256],
            "nhead": [4, 8],
            # Continuous ranges: (min, max) - integers if both are int, float otherwise
            "num_layers": (2, 6),
            "dropout": (0.1, 0.3),
            "memory_size": [128, 256, 512],
            "learning_rate": (1e-5, 1e-3),  # Log scale automatically detected for wide ranges
        },
        "Transformer": {
            "d_model": [64, 128, 256],
            "nhead": [4, 8],
            "num_layers": (2, 6),
            "dropout": (0.1, 0.3),
            "learning_rate": (1e-5, 1e-3),
        }
    },
    "training": {
        # Maximum number of training epochs
        "epochs": 100,
        # Learning rate for optimizer
        "learning_rate": 1e-4,
        # Number of epochs without improvement before stopping training early
        "early_stopping_patience": 10,
        # Whether to enable fine-tuning mode (uses different learning rate)
        "fine_tune": False,
        # Learning rate to use during fine-tuning (typically lower than training LR)
        "fine_tune_learning_rate": 1e-5,
        # Whether to freeze encoder layers during fine-tuning
        "freeze_encoder": False,
        # Number of best model checkpoints to keep per model type
        "max_checkpoints_per_model": 1,
        # Directory to save model checkpoints
        "checkpoint_dir": "ml/checkpoints",
    },
    
    "optimization": {
        # Number of trials for hyperparameter optimization per model
        "n_trials": 1,
        # List of model types to optimize. Use ["all"] to optimize all models with hyperparameter configs
        "models_to_optimize": ["all"],
        # Metric to optimize during hyperparameter search
        # Options: "Loss", "MAE", "MSE", "RMSE", "MAPE", "R2"
        "optimization_metric": "MAPE",
        # Direction of optimization: "minimize" or "maximize"
        "optimization_direction": "minimize",
    },
    
    "evaluation": {
        # Whether to average metrics across the sequence dimension for seq2seq predictions
        # If True and prediction_mode is seq2seq, metrics are averaged across all predicted time steps
        # If False, metrics are computed on the flattened predictions
        "average_sequence_metrics": True,
    }
}

# Auto-detect optimal device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

ml_settings["system"].update({"device": device})
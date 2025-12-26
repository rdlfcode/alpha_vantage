import torch

ml_settings = {
    "system": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
    },
    "data": {
        "target_col": "close",
        "time_col": "dt",
        "group_col": "symbol",
        "sequence_length": 60,
        "prediction_horizon": 1, # predict 1 step ahead
        "train_split": 0.8,
        "val_split": 0.1, # test split is remainder
        "batch_size": 32,
        "num_workers": 0,
        "categorical_cols": [], # Auto-detect if empty
        "numerical_cols": [], # Auto-detect if empty
    },
    "model": {
        "type": "TITANS", # Options: "Linear", "Exponential", "FFT", "Transformer", "TITANS"
        "input_dim": None, # Will be set dynamically based on dataset
        "output_dim": 1,
        # Transformer / TITANS specific
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        # TITANS specific
        "memory_type": "neural", # "neural", "lstm", "transformer"
        "memory_size": 128,
    },
    "training": {
        "epochs": 100,
        "learning_rate": 1e-3,
        "early_stopping_patience": 10,
        "fine_tune": False, # If True, enables specific fine-tuning logic
        "fine_tune_learning_rate": 1e-4,
        "freeze_encoder": False, # Option to freeze encoder during fine-tuning
    }
}

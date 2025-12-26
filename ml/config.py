"""
Configuration for ML models and training.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DataConfig:
    target_col: str
    time_col: str = "dt"
    group_col: str = "symbol"
    sequence_length: int = 60
    prediction_horizon: int = 90
    train_split: float = 0.8
    val_split: float = 0.1
    batch_size: int = 32

@dataclass
class TitansConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    memory_type: str = "neural" # "neural" or "lstm" or "transformer" (if we want to experiment) - for TITANS specifically it usually implies a neural memory
    memory_size: int = 128
    
@dataclass
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    device: str = "cpu" # or "cuda" or "mps"


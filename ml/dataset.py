"""
Dataset implementation with dynamic encoding for AlgoEvals.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from ml.settings import ml_settings

class AlgoEvalsDataset(Dataset):
    """
    Dataset class that dynamically handles numerical and categorical columns.
    It expects a DataFrame and automatically detects column types if not provided.
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        sequence_length: int = None,
        prediction_horizon: int = None,
        prediction_mode: str = None,
        group_col: str = None,
        time_col: str = None, 
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        encoders: Optional[Dict[str, Any]] = None,
        scaler_type: str = "standard",
        scaler_overrides: Optional[Dict[str, str]] = None,
        is_inference: bool = False
    ):
        self.target_col = target_col or ml_settings["data"]["target_col"]
        self.sequence_length = sequence_length or ml_settings["data"]["sequence_length"]
        self.prediction_horizon = prediction_horizon or ml_settings["data"]["prediction_horizon_training"]
        self.prediction_mode = prediction_mode or ml_settings["data"]["prediction_mode"]
        self.group_col = group_col or ml_settings["data"]["group_col"]
        self.time_col = time_col or ml_settings["data"]["time_col"]
        self.scaler_type = scaler_type
        self.scaler_overrides = scaler_overrides or {}
        self.is_inference = is_inference

        # Sort by group and time
        if self.time_col and self.time_col in data.columns:
            self.data = data.sort_values(by=[self.group_col, self.time_col]).reset_index(drop=True)
        elif self.group_col and self.group_col in data.columns:
            self.data = data.sort_values(by=[self.group_col]).reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)

        # Dynamic Column Detection
        if categorical_cols is None or numerical_cols is None:
            self.cat_cols, self.num_cols = self._detect_column_types(self.data)
        else:
            self.cat_cols = categorical_cols
            self.num_cols = numerical_cols
            
        # Ensure target is in num_cols if it's numerical (usually is for regression)
        # If target is categorical, this logic needs adjustment, assuming regression for now as per AlgoEvals description.
        if self.target_col in self.num_cols and self.target_col not in self.cat_cols:
             pass # All good
        
        # Scaling and Encoding
        self.scalers = scalers if scalers else {}
        self.encoders = encoders if encoders else {}
        
        self.processed_data = self._preprocess(self.data.copy())
        
        # Create sequences
        self.sequences = self._create_sequences()

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cat_cols = []
        num_cols = []
        
        exclude_cols = [self.time_col, self.group_col] # Don't use these as features usually, or handle group separately

        for col in df.columns:
            if col in exclude_cols:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for low cardinality to treat as categorical? 
                # For now, treat all numeric as numeric unless specified otherwise.
                num_cols.append(col)
            else:
                cat_cols.append(col)
                
        return cat_cols, num_cols

    def _create_scaler(self, col: str):
        """Create a scaler based on configuration."""
        # Check for column-specific override
        scaler_type = self.scaler_overrides.get(col, self.scaler_type)
        
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Numerical
        # We can use a single scaler for all num cols or individual. 
        # Using individual for flexibility.
        for col in self.num_cols:
            if col not in self.scalers:
                if self.is_inference:
                     # Warn or handle missing scaler? for now assume we must have it
                     raise ValueError(f"Scaler for {col} not found during inference")
                scaler = self._create_scaler(col)
                # Fit only on valid data
                valid_data = df[col].dropna().values.reshape(-1, 1)
                if scaler is not None and len(valid_data) > 0:
                    scaler.fit(valid_data)
                elif scaler is None:
                    pass  # No scaling
                else:
                    print(f"WARNING: Scaler for {col} not fitted! valid_data len is 0. DF len: {len(df)}")
                    print(f"Head of failing col: {df[col].head()}")
                self.scalers[col] = scaler
            
            # Transform
            # Fill NA before scaling? Strategy: ffill then fillna(0)
            df[col] = df[col].ffill().fillna(0)
            
            # Apply scaling if scaler exists
            if self.scalers[col] is not None:
                df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1))

        # Categorical
        for col in self.cat_cols:
            df[col] = df[col].fillna("UNKNOWN")
            if col not in self.encoders:
                if self.is_inference:
                     # Handle unseen categories?
                    le = LabelEncoder()
                    le.fit(["UNKNOWN"]) # Placeholder if really strictly inference
                else:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                self.encoders[col] = le
            
            # Safe transform?
            # For simplicity, map unknown to a default or crash. 
            # Using map/lambda for safety if desired, but standard transform for now.
            # df[col] = self.encoders[col].transform(df[col].astype(str))
            
            # Robust transform for unseen labels
            le = self.encoders[col]
            df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else "UNKNOWN") 
            # If UNKNOWN was not in classes, we have an issue. 
            # Re-fitting UNKNOWN if needed? 
            # Let's simple assume train set covers it or we fallback to 0.
            # Simplified:
            df[col] = df[col].apply(lambda x: self._safe_encode(le, x))

        return df
    
    def _safe_encode(self, le, x):
        try:
            return le.transform([x])[0]
        except ValueError:
            # Fallback to first class or a specific unknown index if we had one
            return 0 

    def _create_sequences(self):
        sequences = []
        # Group by symbol
        grouped = self.processed_data.groupby(self.group_col)
        
        for symbol, group in grouped:
            values = group[self.num_cols].values
            cats = group[self.cat_cols].values if self.cat_cols else None
            targets = group[self.target_col].values
            
            # Need (seq_len + prediction_horizon) length at least
            total_len = len(group)
            if total_len <= self.sequence_length + self.prediction_horizon:
                continue
            
            # Determine how many sequences we can create
            if self.prediction_mode == 'seq2seq':
                # For seq2seq: input [t-seq:t], target [t+1:t+1+horizon]
                max_i = total_len - self.sequence_length - self.prediction_horizon
            else:
                # For seq2point: input [t-seq:t], target [t+horizon]
                max_i = total_len - self.sequence_length - self.prediction_horizon + 1
            
            for i in range(max_i):
                # Inputs
                x_num = values[i : i + self.sequence_length]
                x_cat = cats[i : i + self.sequence_length] if cats is not None else []
                
                # Target
                if self.prediction_mode == 'seq2seq':
                    # Predict a sequence of future values
                    y = targets[i + self.sequence_length : i + self.sequence_length + self.prediction_horizon]
                else:
                    # Predict a single point in the future (seq2point)
                    y = targets[i + self.sequence_length + self.prediction_horizon - 1]
                
                sequences.append({
                    'x_num': torch.tensor(x_num, dtype=torch.float32),
                    'x_cat': torch.tensor(x_cat, dtype=torch.long) if cats is not None else torch.empty(0),
                    'y': torch.tensor(y, dtype=torch.float32)
                })
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def get_feature_dims(self):
        return len(self.num_cols), len(self.cat_cols)
        
    def get_cat_dims(self):
        return [len(self.encoders[col].classes_) for col in self.cat_cols]


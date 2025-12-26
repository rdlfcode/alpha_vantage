"""
Dataset implementation with dynamic encoding for AlgoEvals.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

class AlgoEvalsDataset(Dataset):
    """
    Dataset class that dynamically handles numerical and categorical columns.
    It expects a DataFrame and automatically detects column types if not provided.
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        group_col: str = "symbol",
        time_col: str = "dt", 
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        encoders: Optional[Dict[str, Any]] = None,
        is_inference: bool = False
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.group_col = group_col
        self.time_col = time_col
        self.is_inference = is_inference

        # Sort by group and time
        if time_col in data.columns:
            self.data = data.sort_values(by=[group_col, time_col]).reset_index(drop=True)
        else:
            self.data = data.sort_values(by=[group_col]).reset_index(drop=True)

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

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Numerical
        # We can use a single scaler for all num cols or individual. 
        # Using individual for flexibility.
        for col in self.num_cols:
            if col not in self.scalers:
                if self.is_inference:
                     # Warn or handle missing scaler? for now assume we must have it
                     raise ValueError(f"Scaler for {col} not found during inference")
                scaler = StandardScaler()
                # Fit only on valid data
                valid_data = df[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    scaler.fit(valid_data)
                self.scalers[col] = scaler
            
            # Transform
            # Fill NA before scaling? Strategy: ffill then fillna(0)
            df[col] = df[col].ffill().fillna(0) 
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
            # Actually, for training: Input [t-seq : t], Target [t+horizon]
            
            total_len = len(group)
            if total_len <= self.sequence_length + self.prediction_horizon:
                continue
                
            for i in range(total_len - self.sequence_length - self.prediction_horizon + 1):
                # Inputs
                x_num = values[i : i + self.sequence_length]
                x_cat = cats[i : i + self.sequence_length] if cats is not None else []
                
                # Target
                # Predict the value at i + seq_length + horizon - 1?
                # or the sequence of future values?
                # User asked for "Prediction (n points into the future)"
                # Let's predict the single point at horizon for simplicity first, or the sequence.
                # Usually seq-to-seq or seq-to-point. 
                # Let's do seq-to-point for now.
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


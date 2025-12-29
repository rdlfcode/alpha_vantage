import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ml.settings import ml_settings
from ml.dataset import AlgoEvalsDataset
from ml.models import create_model
from ml.train import train_model

def create_dummy_data(n_samples=1000):
    dates = pd.date_range(start="2023-01-01", periods=n_samples)
    data = pd.DataFrame({
        "dt": dates,
        "symbol": ["TEST"] * n_samples,
        "close": np.sin(np.linspace(0, 100, n_samples)) + np.random.normal(0, 0.1, n_samples),
        "volume": np.random.randint(100, 1000, n_samples)
    })
    return data

def test_generic_training():
    print("--- Testing Generic Training ---")
    
    # 1. Setup Data
    df = create_dummy_data()
    
    # Update settings for test
    ml_settings["data"]["target_col"] = "close"
    ml_settings["data"]["sequence_length"] = 10
    ml_settings["data"]["prediction_horizon"] = 1
    ml_settings["model"]["input_dim"] = 1 # Will be updated after dataset creation
    ml_settings["training"]["epochs"] = 2
    ml_settings["system"]["device"] = "cpu"
    
    # Create Dataset
    dataset = AlgoEvalsDataset(
        data=df,
        target_col=ml_settings["data"]["target_col"],
        sequence_length=ml_settings["data"]["sequence_length"],
        prediction_horizon=ml_settings["data"]["prediction_horizon"],
        time_col=ml_settings["data"]["time_col"],
        group_col=ml_settings["data"]["group_col"]
    )
    
    # Update input_dim in settings based on dataset
    num_features, _ = dataset.get_feature_dims()
    ml_settings["model"]["input_dim"] = num_features
    print(f"Input Dim detected: {num_features}")
    
    # Create Loader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Create Model
    # Test TITANS
    ml_settings["model"]["type"] = "TITANS"
    model = create_model(ml_settings)
    print(f"Created model: {type(model).__name__}")
    
    # 3. Train
    print("Running training...")
    history = train_model(model, loader, loader, ml_settings) # using same loader for val for simplicity
    print("Training finished.")
    
    # Verify metrics in history
    last_hist = history[-1]
    required_metrics = ["MAE", "MSE", "RMSE", "R2", "Adj_R2", "MAPE", "RMSLE"]
    for m in required_metrics:
        if m in last_hist:
            print(f"Metric {m}: {last_hist[m]:.4f}")
        else:
            print(f"ERROR: Metric {m} missing!")

    # 4. Test Fine-tuning
    print("\n--- Testing Fine-tuning ---")
    ml_settings["training"]["fine_tune"] = True
    ml_settings["training"]["epochs"] = 1
    ml_settings["training"]["freeze_encoder"] = True
    
    print("Running fine-tuning...")
    history_ft = train_model(model, loader, loader, ml_settings)
    print("Fine-tuning finished.")
    
    assert len(history) == 2
    assert len(history_ft) == 1
    print("\nSUCCESS: All tests passed!")

if __name__ == "__main__":
    test_generic_training()

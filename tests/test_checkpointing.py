
import os
import torch
import shutil
import duckdb
import json
import logging
from torch.utils.data import DataLoader, TensorDataset
from ml.train import train_model
from data.alpha_vantage_schema import TABLE_SCHEMAS

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {
            "x_num": self.x[idx],
            "x_cat": torch.empty(0), # Dummy existing cat features?
            "y": self.y[idx]
        }

def test_checkpointing():
    print("Testing Checkpointing...")
    
    # 1. Setup Dummy Data and Model
    # x_num needs to mock input dimensions
    # In settings, input_dim is auto set or model handles it.
    # Linear model expects flat input. 
    # train_one_epoch expects batch['x_num']
    
    X = torch.randn(10, 10)
    y = torch.randn(10, 1)
    
    dataset = DictDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)
    
    model = torch.nn.Linear(10, 1)
    
    # 2. Setup Test Environment
    test_db = "test_checkpointing.db"
    checkpoints_dir = "test_checkpoints"
    
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        
    conn = duckdb.connect(test_db)
    # Create schema
    conn.execute(TABLE_SCHEMAS["MODEL_METADATA"])
    
    settings = {
        "system": {"device": "cpu"},
        "data": {},
        "model": {"name": "test_model", "type": "linear"},
        "training": {
            "epochs": 5, 
            "learning_rate": 0.01,
            "max_checkpoints_per_model": 2, # Keep only top 2
            "checkpoint_dir": checkpoints_dir
        }
    }
    
    # 3. Run Training
    print("Running training loop...")
    history = train_model(model, loader, loader, settings, db_conn=conn)
    
    # 4. Verify Filesystem Checkpoints
    print("Verifying filesystem checkpoints...")
    files = list(os.listdir(checkpoints_dir))
    pth_files = [f for f in files if f.endswith(".pth")]
    print(f"Found checkpoints: {pth_files}")
    
    # Expect at most 2 files (since max_checkpoints_per_model=2)
    assert len(pth_files) <= 2, f"Expected <= 2 checkpoints, found {len(pth_files)}"
    assert len(pth_files) > 0, "No checkpoints found!"
    
    # 5. Verify JSON Metadata
    print("Verifying JSON metadata...")
    json_path = os.path.join(checkpoints_dir, "models_metadata.json")
    assert os.path.exists(json_path), "Metadata JSON not found"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        assert "test_model" in data
        assert len(data["test_model"]) > 0
        print("JSON Metadata content validated.")
        
    # 6. Verify DB
    print("Verifying Database...")
    result = conn.execute("SELECT * FROM MODEL_METADATA").df()
    print("DB Content:")
    print(result)
    assert not result.empty, "DB table is empty"
    assert result.iloc[0]["model_name"] == "test_model"
    
    # Clean up
    conn.close()
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        
    print("Test passed successfully!")

if __name__ == "__main__":
    test_checkpointing()

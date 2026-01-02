import os
import json
import torch
import numpy as np
import logging
import duckdb
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from data.alpha_vantage_schema import ENDPOINT_TO_TABLE_MAP, TABLE_SCHEMAS, TABLE_PKS

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages model checkpoints and metadata storage.
    
    Features:
    - Saves model state_dict to filesystem.
    - Maintains only the Top N best checkpoints per model (based on validation loss).
    - Stores comprehensive metadata in a local JSON file (redundancy).
    - Stores metadata in the DuckDB database.
    """
    
    def __init__(
        self, 
        model_name: str, 
        model_type: str,
        checkpoint_dir: str = "ml/checkpoints", 
        settings: Dict[str, Any] = None,
        db_conn: Optional[duckdb.DuckDBPyConnection] = None
    ):
        """
        Args:
            model_name: Unique name for the model (e.g., "titans_v1").
            model_type: Type of model (e.g., "TITANS", "Transformer").
            checkpoint_dir: Directory to store checkpoints.
            settings: Settings dictionary containing 'max_checkpoints_per_model'.
            db_conn: DuckDB connection.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.settings = settings or {}
        # Default to 3 if not set
        self.max_checkpoints = self.settings.get("training", {}).get("max_checkpoints_per_model", 3)
        self.model_settings = self.settings.get("model", {})
        
        self.db_conn = db_conn
        self.metadata_file = self.checkpoint_dir / "models_metadata.json"
        
        # Unique ID for this training run
        self.run_id = str(uuid.uuid4())
        
        logger.info(f"Initialized CheckpointManager for {model_name} (Run ID: {self.run_id}). Max checkpoints: {self.max_checkpoints}")

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert numpy types to python native types for JSON serialization."""
        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                sanitized[k] = v.item()
            elif isinstance(v, torch.Tensor):
                sanitized[k] = v.item()
            else:
                sanitized[k] = v
        return sanitized

    def save(self, model: torch.nn.Module, metrics: Dict[str, float], dataset_size: int, epoch: int):
        """
        Orchestrates saving of checkpoint and metadata.
        
        Args:
            model: The PyTorch model to save.
            metrics: Dictionary of metrics (must include 'Loss' or 'val_loss').
            dataset_size: Size of the training dataset.
            epoch: Current epoch number.
        """
        val_loss = metrics.get("Loss")
        if val_loss is None:
            logger.warning("No 'Loss' found in metrics. Cannot rank checkpoint.")
            return

        # 1. Save Checkpoint to Filesystem
        checkpoint_filename = f"{self.model_name}_run_{self.run_id}_epoch_{epoch}_loss_{val_loss:.6f}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        try:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return

        # Sanitize metrics
        safe_metrics = self._sanitize_metrics(metrics)

        # 2. Manage Top N Checkpoints (Delete old ones)
        self._manage_filesystem_checkpoints(checkpoint_path, val_loss)

        # 3. Create Metadata Dict
        metadata = {
            "dt": datetime.utcnow().isoformat(),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "training_run_id": self.run_id,
            "epoch": epoch,
            "metrics": safe_metrics,
            "best_val_loss": float(val_loss),
            "dataset_size": dataset_size,
            "checkpoint_path": str(checkpoint_path.absolute()),
            "model_config": self.model_settings
        }

        # 4. Save Metadata to JSON
        self._save_metadata_json(metadata)

        # 5. Save Metadata to DB
        self._save_metadata_db(metadata)

    def _manage_filesystem_checkpoints(self, new_checkpoint_path: Path, new_val_loss: float):
        """
        Ensures only the top N checkpoints for this model exist.
        """
        # Get all checkpoints for this model name
        # Pattern: {model_name}_run_...
        # We need a robust way to track them. 
        # Ideally, we read the JSON metadata to find existing checkpoints, or scan the dir.
        # Scanning dir is safer if JSON is corrupted.
        
        files = list(self.checkpoint_dir.glob(f"{self.model_name}_*.pth"))
        
        # Parse files to extract loss
        # Filename format: {model_name}_run_{run_id}_epoch_{epoch}_loss_{loss}.pth
        checkpoints = []
        for f in files:
            try:
                # Extract loss from filename (last underscore before .pth)
                parts = f.stem.split("_loss_")
                if len(parts) > 1:
                    loss = float(parts[-1])
                    checkpoints.append((loss, f))
            except Exception:
                continue
                
        # Include the new one if it wasn't picked up (it should be there)
        if (new_val_loss, new_checkpoint_path) not in checkpoints:
             checkpoints.append((new_val_loss, new_checkpoint_path))
             
        # Sort by loss (ascending, lower is better)
        checkpoints.sort(key=lambda x: x[0])
        
        # Keep top N
        if len(checkpoints) > self.max_checkpoints:
            to_delete = checkpoints[self.max_checkpoints:]
            for _, path in to_delete:
                try:
                    if path.exists():
                        os.remove(path)
                        logger.info(f"Deleted old checkpoint: {path.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {path}: {e}")

    def _save_metadata_json(self, metadata: Dict[str, Any]):
        """
        Appends or updates metadata in a JSON file.
        """
        try:
            full_data = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    try:
                        full_data = json.load(f)
                    except json.JSONDecodeError:
                        full_data = {}
            
            if self.model_name not in full_data:
                full_data[self.model_name] = []
            
            # Append this run/checkpoint info
            full_data[self.model_name].append(metadata)
            
            with open(self.metadata_file, "w") as f:
                json.dump(full_data, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to save metadata to JSON: {e}")

    def _save_metadata_db(self, metadata: Dict[str, Any]):
        """
        Upserts metadata into DuckDB MODEL_METADATA table.
        """
        if not self.db_conn:
            return

        try:
            # Prepare row
            row = {
                "dt": metadata["dt"],
                "model_name": metadata["model_name"],
                "model_type": metadata["model_type"],
                "training_run_id": metadata["training_run_id"],
                "best_val_loss": metadata["best_val_loss"],
                "dataset_size": metadata["dataset_size"],
                "checkpoint_path": metadata["checkpoint_path"],
            }
            
            # Unpack metrics
            metrics = metadata["metrics"]
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
                
            for m in ["MAE", "MSE", "RMSE", "MAPE", "R2", "Adj_R2", "RMSLE", "Loss"]:
                if m in metrics:
                    row[m] = metrics[m]
                else:
                    row[m] = None
            
            df = pd.DataFrame([row])
            
            # Insert into DB
            table_name = "MODEL_METADATA"

            self.db_conn.execute(f"INSERT OR REPLACE INTO {table_name} BY NAME SELECT * FROM df")
            logger.info(f"Saved metadata to DB for run {self.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata to DB: {e}")

import copy
import optuna
from ml.settings import ml_settings
from ml.train import train_model, create_model
from ml.dataset import AlgoEvalsDataset
from torch.utils.data import DataLoader
from build_dataset import build_dataset

# Configure logging for Optuna to show less output if desired
optuna.logging.set_verbosity(optuna.logging.INFO)

def suggest_parameter(trial, name, value_config):
    """
    Helper to call the right trial.suggest_* method based on config value type.
    """
    # List -> Categorical
    if isinstance(value_config, list):
        return trial.suggest_categorical(name, value_config)
    
    # Tuple -> Range
    if isinstance(value_config, tuple) and len(value_config) == 2:
        start, end = value_config
        
        # Check if int or float
        if isinstance(start, int) and isinstance(end, int):
            return trial.suggest_int(name, start, end)
        else:
            # We can guess log scale for learning rate if the range is large (like 1e-5 to 1e-3)
            # A simple heuristic: if end/start > 100, use log
            log = False
            if start > 0 and end > 0 and (end / start) >= 100:
                log = True
            return trial.suggest_float(name, start, end, log=log)
            
    # Fallback: Just return the value
    return value_config

def update_settings_from_trial(settings, trial, hyperparams_config):
    """
    Updates a deep copy of settings with suggested params from the trial.
    """
    new_settings = copy.deepcopy(settings)
    
    # Map for flat param keys to likely locations in settings
    # This is a bit specific to our valid keys
    # Now we look up based on the model type params
    
    training_keys = ["learning_rate", "batch_size", "epochs"]
    model_type = new_settings['model']['type']
    
    params = {}
    for key, config in hyperparams_config.items():
        val = suggest_parameter(trial, key, config)
        params[key] = val
        
        if key in training_keys:
            new_settings['training'][key] = val
        else:
            # Assume it is a model param
            if 'params' not in new_settings['model']:
                new_settings['model']['params'] = {}
            
            if model_type not in new_settings['model']['params']:
                new_settings['model']['params'][model_type] = {}
                
            new_settings['model']['params'][model_type][key] = val
            
    return new_settings, params

def run_optimization(n_trials=20):
    print(f"Starting Optuna Optimization with {n_trials} trials...")
    
    model_type = ml_settings['model']['type']
    if model_type not in ml_settings.get('hyperparameters', {}):
        print(f"No hyperparameter configuration found for model type: {model_type}")
        return

    hp_config = ml_settings['hyperparameters'][model_type]
    
    # 1. Prepare Data (Once)
    print("Loading dataset...")
    df = build_dataset()
    
    feature_cols = ml_settings['data'].get('feature_columns', ['all'])
    if 'all' not in feature_cols:
         cols_to_keep = list(set(feature_cols + [ml_settings['data']['target_col'], ml_settings['data']['time_col'], ml_settings['data']['group_col']]))
         df = df[cols_to_keep]

    train_split = ml_settings['data']['train_split']
    time_col = ml_settings['data']['time_col']
    df = df.sort_values(by=[time_col]).reset_index(drop=True)
    
    unique_times = df[time_col].unique()
    n_unique = len(unique_times)
    train_end_idx = int(n_unique * train_split)
    train_cutoff = unique_times[train_end_idx]
    
    train_df = df[df[time_col] < train_cutoff].copy()
    val_df = df[df[time_col] >= train_cutoff].copy()
    
    # Initial Scaler Fit
    train_dataset = AlgoEvalsDataset(
        train_df, 
        target_col=ml_settings['data']['target_col'],
        sequence_length=ml_settings['data']['sequence_length'],
        prediction_horizon=ml_settings['data']['prediction_horizon']
    )
    val_dataset = AlgoEvalsDataset(
        val_df, 
        target_col=ml_settings['data']['target_col'],
        sequence_length=ml_settings['data']['sequence_length'],
        prediction_horizon=ml_settings['data']['prediction_horizon'],
        scalers=train_dataset.scalers, 
        encoders=train_dataset.encoders 
    )

    num_vals, cat_vals = train_dataset.get_feature_dims()
    input_dim = num_vals
    
    def objective(trial):
        # 2. Setup Settings for this Trial
        trial_settings, current_params = update_settings_from_trial(ml_settings, trial, hp_config)
        trial_settings['model']['input_dim'] = input_dim
        
        # 3. Create Model
        try:
            model = create_model(trial_settings)
        except Exception as e:
            # Prune invalid models immediately (e.g. invalid config)
            print(f"Trial failed to create model: {e}")
            raise optuna.TrialPruned()

        # 4. Data Loaders
        batch_size = trial_settings['data']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 5. Train
        # We define a custom pruning check if we want, but for now we'll just run it.
        # Ideally, we would pass 'trial' into train_model to report intermediate values.
        # For simplicity, we run the full training and return the final val metric.
        
        history = train_model(model, train_loader, val_loader, trial_settings, db_conn=None)
        
        if not history:
             raise optuna.TrialPruned()
            
        # Optimize for Validation Loss (last epoch or best epoch?)
        # Dictionary has 'Loss' which is val loss
        best_val_loss = min([h['Loss'] for h in history])
        
        return best_val_loss

    # Create Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n==============================")
    print("Optimization Complete")
    print(f"Best Loss: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==============================")
    
    return study.best_params

if __name__ == "__main__":
    # Example usage:
    # Adjust settings or epochs as needed before running
    # ml_settings['training']['epochs'] = 50 
    run_optimization(n_trials=20)

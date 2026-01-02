import duckdb
import logging
from pathlib import Path
from data.settings import settings
from data.alpha_vantage import AlphaVantageClient
from data.alpha_vantage_schema import ENDPOINT_TO_TABLE_MAP, TABLE_SCHEMAS, DEFAULT_ENDPOINTS

# ML imports
from ml.algo_evals import AlgoEvals
from ml.settings import ml_settings

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicates/console pollution
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Add FileHandler
file_handler = logging.FileHandler("av.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("Starting Alpha Vantage Data Update and ML Pipeline")
    logger.info("=" * 80)
    
    # ========================================================================
    # Step 1: Update Data
    # ========================================================================
    logger.info("\n[1/3] Updating data from Alpha Vantage...")
    
    # Ensure tables exist
    db_path = Path(settings.get("data_dir"), settings.get("db_name"))
    conn = duckdb.connect(str(db_path), read_only=False)
    tables = set()

    for endpoint_name in DEFAULT_ENDPOINTS:
        tables.add(ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper())

    for table_name in tables:
        schema_sql = TABLE_SCHEMAS.get(table_name)
        if schema_sql:
            create_sql = schema_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
            conn.execute(create_sql)

    client = AlphaVantageClient(db_conn=conn)

    # Update data efficiently
    client.smart_update(
        symbols=['GOOG'],
        endpoints=DEFAULT_ENDPOINTS,
        start_date='2010-12-20',
        end_date='2025-12-19'
    )
    
    # Close the client's connection but keep our main connection
    conn.close()
    
    logger.info("Data update complete!")
    
    # ========================================================================
    # Step 2: Train and Optimize Models
    # ========================================================================
    logger.info("\n[2/3] Training and optimizing ML models...")
    
    # Create AlgoEvals instance
    with AlgoEvals(ml_settings) as algo_evals:
        # Load data
        algo_evals.load_data()
        
        # Prepare datasets
        algo_evals.prepare_datasets()
        
        # Train all models with hyperparameter optimization
        logger.info(f"\nOptimizing models: {ml_settings['optimization']['models_to_optimize']}")
        logger.info(f"Prediction mode: {ml_settings['data']['prediction_mode']}")
        logger.info(f"Training horizon: {ml_settings['data']['prediction_horizon_training']} steps")
        logger.info(f"Inference horizon: {ml_settings['data']['prediction_horizon_inference']} days\n")
        
        results = algo_evals.train_all_models(optimize=True)
        
        # ========================================================================
        # Step 3: Generate Predictions
        # ========================================================================
        logger.info("\n[3/3] Generating future predictions...")
        
        # Use the best model from results
        if results:
            best_result = max(results, key=lambda r: r.get('optimization', {}).get('best_value', float('inf')))
            best_model = best_result['model']
            best_model_type = best_result['model_type']
            
            logger.info(f"\nUsing {best_model_type} for predictions")
            logger.info(f"Best validation metric: {best_result.get('best_val_loss', 'N/A'):.4f}")
            
            # Generate 91-day predictions
            predictions = algo_evals.generate_predictions(
                model=best_model,
                model_name=best_model_type,
                horizon_days=91
            )
            
            logger.info(f"\nGenerated {len(predictions)} predictions")
            if len(predictions) > 0:
                logger.info(f"Sample predictions:\n{predictions.head(10)}")
        else:
            logger.warning("No models were trained successfully!")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
   main()

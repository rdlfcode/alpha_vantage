import logging
import pandas as pd
from alpha_vantage import AlphaVantageClient
import time

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
    logger.info("Starting Alpha Vantage Data Updater...")
    client = AlphaVantageClient()

    # Update data
    start_time = time.time()
    df = client.get_data(
        # We want EOD data
        end_date=(pd.to_datetime(start_time) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    )
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Update completed in {duration:.2f} seconds.")
    
    if not df.empty:
        logger.info(f"Fetched {len(df)} new rows of data.")
    else:
        logger.info("No new data was fetched (everything up to date).")

if __name__ == "__main__":
   main()

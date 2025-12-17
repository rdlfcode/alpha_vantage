
import duckdb
import logging
from alpha_vantage import AlphaVantageClient
from utils import read_stock_symbols
import time

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Checks if handlers are already configured (e.g. by alpha_vantage)
# We want to ensure FileHandler is present
has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
if not has_file_handler:
    file_handler = logging.FileHandler("av.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)

# Ensure StreamHandler (or TqdmHandler) is present
# alpha_vantage adds TqdmLoggingHandler, so we might not need StreamHandler if that's there.
# But just in case:
has_console_handler = any(isinstance(h, (logging.StreamHandler, logging.FileHandler)) for h in root_logger.handlers) 
# Note: FileHandler is a StreamHandler subclass? No.
# Actually TqdmLoggingHandler is a Handler.
# Let's just add StreamHandler if no handlers exist at all?
# No, alpha_vantage adds one.
# So we are good with just adding FileHandler.

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Alpha Vantage Data Updater...")
    from alpha_vantage_schema import ENDPOINT_TO_TABLE_MAP
    logger.info(f"Map for INCOME_STATEMENT: {ENDPOINT_TO_TABLE_MAP.get('INCOME_STATEMENT')}")
    client = AlphaVantageClient()

    # Load symbols
    symbols = read_stock_symbols()
    logger.info(f"Loaded {len(symbols)} symbols.")

    # Update data
    start_time = time.time()
    df = client.get_data()
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Update completed in {duration:.2f} seconds.")
    
    if not df.empty:
        logger.info(f"Fetched {len(df)} new rows of data.")
    else:
        logger.info("No new data was fetched (everything up to date).")

if __name__ == "__main__":
   main()

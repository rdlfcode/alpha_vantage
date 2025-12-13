import time
import logging
from collections import deque
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from alpha_vantage import AlphaVantageClient
from alpha_vantage_schema import DEFAULT_ENDPOINTS, SYMBOL_ENDPOINTS
from utils import read_stock_symbols

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting data consumption script...")

    # Initialize Client
    try:
        client = AlphaVantageClient()
    except ValueError as e:
        logger.error(f"Initialization Error: {e}")
        return

    # Load Symbols
    symbols = read_stock_symbols()
    logger.info(f"Loaded {len(symbols)} symbols: {symbols}")

    # Build Task List
    tasks = deque()
    
    # Prioritize endpoints order if needed, otherwise random
    for endpoint_name, params in DEFAULT_ENDPOINTS.items():
        if endpoint_name in SYMBOL_ENDPOINTS:
            for symbol in symbols:
                # Create specific params for this symbol
                task_params = params.copy()
                task_params['symbol'] = symbol
                tasks.append((endpoint_name, task_params))
        else:
            # Macro endpoint, run once
            tasks.append((endpoint_name, params))

    logger.info(f"Generated {len(tasks)} tasks.")

    # Process Tasks
    while tasks:
        endpoint_name, params = tasks[0] # Peek
        
        task_desc = f"{endpoint_name}"
        if 'symbol' in params:
            task_desc += f" ({params['symbol']})"
            
        logger.info(f"Processing task: {task_desc}")

        try:
            # We use fetch_and_cache_data directly to have fine-grained control
            # This method handles DB and Parquet saving
            df = client.fetch_and_cache_data(
                endpoint_name,
                params,
                True)
            
            if df.empty:
                logger.warning(f"No data returned for {task_desc}")
            else:
                logger.info(f"Success: {task_desc} - {len(df)} rows")
            
            # Remove completed task
            tasks.popleft()

        except Exception as e:
            msg = str(e)
            if "Daily request limit reached" in msg:
                logger.warning(f"Daily limit hit while processing {task_desc}.")
                
                # Calculate sleep time
                # We need to wait until the oldest request in the daily window expires
                # RateLimiter stores timestamps in self.day_timestamps
                # It is a deque of floats (timestamps)
                
                timestamps = client.rate_limiter.day_timestamps
                if timestamps:
                    oldest = timestamps[0]
                    now = time.time()
                    # Reset time is 24h after oldest
                    reset_time = oldest + (24 * 60 * 60)
                    wait_seconds = reset_time - now
                    
                    if wait_seconds < 0:
                         # Should not happen if limit is reached, but safety first
                         wait_seconds = 60
                         
                    # Add a small buffer safely
                    wait_seconds += 60 
                    
                    hours = wait_seconds / 3600
                    logger.info(f"Sleeping for {wait_seconds:.0f} seconds ({hours:.2f} hours) until rate limit resets.")
                    
                    time.sleep(wait_seconds)
                    logger.info("Resuming...")
                    # Do not pop task, retry it
                else:
                    logger.error("Daily limit reported but no timestamps found? Waiting 1 hour fallback.")
                    time.sleep(3600)
            else:
                logger.error(f"Unexpected error processing {task_desc}: {e}")
                # decide whether to skip or retry. skipping to avoid infinite loop on bad params
                tasks.popleft()

    logger.info("All tasks completed.")

if __name__ == "__main__":
    main()

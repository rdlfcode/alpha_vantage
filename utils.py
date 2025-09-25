"""
Utility functions for Alpha Vantage data processing.
"""

import pandas as pd
import duckdb
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def read_stock_symbols(file_path: str = "stocks.txt") -> List[str]:
    """
    Read stock symbols from a text file.
    
    Args:
        file_path: Path to the text file containing stock symbols
        
    Returns:
        List of stock symbols
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            symbols = [line.strip().upper() for line in file if line.strip()]
        logger.info(f"Read {len(symbols)} stock symbols from {file_path}")
        return symbols
    except FileNotFoundError:
        logger.warning(f"Stock symbols file {file_path} not found. Using empty list.")
        return []
    except Exception as e:
        logger.error(f"Error reading stock symbols from {file_path}: {e}")
        return []


def get_dataset(sql_query: str, db_path: str) -> pd.DataFrame:
    """
    Execute SQL query against the database and return results as DataFrame.
    
    Args:
        sql_query: SQL query to execute
        db_path: Path to the database file
        
    Returns:
        Query results as DataFrame
    """
    try:
        conn = duckdb.connect(db_path)
        df = conn.execute(sql_query).df()
        conn.close()
        logger.info(f"Executed query successfully, returned {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame()
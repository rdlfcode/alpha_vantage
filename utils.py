"""
Utility functions for Alpha Vantage data processing.
"""

import pandas as pd
import duckdb
from typing import List, Dict, Optional, Union
import logging
from settings import settings
import alpha_vantage_schema as avs

logger = logging.getLogger(__name__)


def get_default_params(endpoint_name: str) -> dict:
    """
    Get default parameters for a given endpoint from the schema.
    This function simplifies parameter creation by extracting the first valid
    value for parameters that require a specific choice (e.g., interval).
    """
    params = {}
    if endpoint_name in avs.ALPHA_VANTAGE_SCHEMA:
        for param, values in avs.ALPHA_VANTAGE_SCHEMA[endpoint_name].items():
            if isinstance(values, list):
                # Use the first value as the default for list-based params
                params[param] = values[0]
            elif values is None or values == "string":
                # Skip string parameters like 'symbol' which are handled separately
                continue
            else:
                params[param] = values
    return params


def get_endpoints(endpoints: Optional[Union[Dict, List]] = None) -> dict:
    endpoints = endpoints or avs.ALPHA_VANTAGE_SCHEMA
    # Only get premium endpoints if premium account
    if not settings.get("AlphaVantagePremium", False):
        return {name: get_default_params(name) for name in endpoints if name in avs.PREMIUM_ENDPOINTS}
    
    return {name: get_default_params(name) for name in endpoints}


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

def generate_create_table_statement(df: pd.DataFrame, table_name: str) -> str:
    """
    Generate a SQL CREATE TABLE statement from a pandas DataFrame.
    Args:
        df: The DataFrame to infer schema from
        table_name: Name of the SQL table
    Returns:
        CREATE TABLE statement as a string
    """
    dtype_map = {
        'int64': 'INT',
        'float64': 'DECIMAL(20, 4)',
        'object': 'TEXT',
        'datetime64[ns]': 'TIMESTAMP',
        'bool': 'BOOLEAN',
    }
    columns = []
    for col, dtype in df.dtypes.items():
        sql_type = dtype_map.get(str(dtype), 'TEXT')
        columns.append(f"    {col} {sql_type}")
    columns_str = ",\n".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{columns_str}\n);"
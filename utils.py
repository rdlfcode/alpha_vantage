"""
Utility functions for Alpha Vantage data processing.
"""

import pandas as pd
import duckdb
from typing import List, Dict, Optional, Union
import logging
from settings import settings
import alpha_vantage_schema as avs
import numpy as np
from decimal import Decimal, InvalidOperation

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
   endpoints = endpoints or avs.DEFAULT_ENDPOINTS
   # Only get premium endpoints if premium account
   if not settings.get("AlphaVantagePremium", False):
      return {name: get_default_params(name) for name in endpoints if name not in avs.PREMIUM_ENDPOINTS}
    
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

def generate_filepath(data_dir: str, endpoint_name: str, params: Dict) -> Path:
   """
   Generates a consistent filename from the endpoint and its parameters.
   """
   filename_parts = []
   params_copy = params.copy()

   # Handle symbol-like parameters first
   symbol_like_keys = ["symbol", "keywords", "tickers", "from_symbol", "to_symbol"]
   for key in symbol_like_keys:
      if key in params_copy:
            filename_parts.append(f"symbol_{params_copy.pop(key)}")
            break

   # Add other parameters in sorted order
   for key, value in sorted(params_copy.items()):
      if key != "apikey" and isinstance(value, str):
            filename_parts.append(f"{key}_{value}")

   filename = "_".join(filename_parts) + settings.get("data_ext", ".parquet")
   path = data_dir / "files" / endpoint_name / filename

   return path

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

def infer_sql_type(column: pd.Series) -> str:
   """
   Infers the most appropriate SQL data type for a pandas Series.

   This function analyzes the data type and content of a column to determine
   a suitable SQL type, including size, precision, and scale where applicable.

   Args:
      column (pd.Series): The pandas Series (DataFrame column) to analyze.

   Returns:
      str: The inferred SQL data type as a string (e.g., 'VARCHAR(100)', 
          'DECIMAL(10, 2)', 'INTEGER').
   """
   # Drop missing values for analysis, but keep track of their existence
   col_non_null = column.dropna()

   # If the column is empty or all values are null, default to a generic type
   if col_non_null.empty:
      return 'VARCHAR(255)'

   dtype = col_non_null.dtype

   # --- Direct Mapping for Specific dtypes ---
   if pd.api.types.is_integer_dtype(dtype):
      max_val = col_non_null.max()
      if max_val < 32767 and col_non_null.min() > -32768:
         return 'SMALLINT'
      elif max_val < 2147483647 and col_non_null.min() > -2147483648:
         return 'INTEGER'
      else:
         return 'BIGINT'

   if pd.api.types.is_float_dtype(dtype):
      # Determine precision and scale for decimal types
      max_precision = 0
      max_scale = 0
      for val in col_non_null:
         try:
            # Use Decimal for precise representation
            d_val = Decimal(str(val))
            if d_val.is_finite():
               sign, digits, exponent = d_val.as_tuple()
               exponent = int(exponent)
               scale = -exponent
               precision = len(digits)
                    
               if scale < 0: # Handle scientific notation like 1.2E+7
                  precision -= scale
                  scale = 0

               max_precision = max(max_precision, precision)
               max_scale = max(max_scale, scale)
         except InvalidOperation:
            # Handle non-finite values like 'inf'
            return 'FLOAT'
      # Ensure precision is at least as large as scale
      max_precision = max(max_precision, max_scale)
      if max_precision == 0: # If all values are 0.0
         return 'DECIMAL(1, 0)'
      return f'DECIMAL({max_precision}, {max_scale})'

   if pd.api.types.is_bool_dtype(dtype):
      return 'BOOLEAN'

   if pd.api.types.is_datetime64_any_dtype(dtype):
      return 'TIMESTAMP'

   # --- Complex Parsing for 'object' dtype ---
   if pd.api.types.is_object_dtype(dtype):
      # Attempt to convert to a more specific type
        
      # 1. Try numeric conversion
      col_numeric = pd.to_numeric(col_non_null, errors='coerce')
      if not col_numeric.isnull().all():
         # Check if it was converted to integer or float
         if (col_numeric % 1 == 0).all(): # All are whole numbers
            return infer_sql_type(col_numeric.astype(np.int64))
         else: # Contains decimals
            return infer_sql_type(col_numeric)

      # 2. Try datetime conversion
      try:
         col_datetime = pd.to_datetime(col_non_null, errors='coerce')
         if not col_datetime.isnull().all():
            return 'TIMESTAMP'
      except Exception:
         pass

      # 3. If all conversions fail, treat as string
      max_len = int(col_non_null.astype(str).str.len().max())
      # Use a sensible default if max_len is 0 (e.g., column of empty strings)
      if max_len == 0:
         return 'VARCHAR(1)'
      # Add a small buffer and round up to a reasonable number
      # for future data that might be slightly longer.
      varchar_size = min(max(32, int(max_len * 1.2)), 8000)
      return f'VARCHAR({varchar_size})'

   # Fallback for any other unhandled type
   return 'VARCHAR(255)'

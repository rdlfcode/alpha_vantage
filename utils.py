"""
Utility functions for Alpha Vantage data processing.
"""
import pandas as pd
import duckdb
from duckdb import DuckDBPyConnection
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
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

def get_latest_date_from_db(conn: DuckDBPyConnection, endpoint_name: str, params: Dict) -> Optional[datetime]:
    """
    Gets the latest date available for a given endpoint/symbol from the DB.
    """
    table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()

    if table_name == "MACRO":
        col_name = endpoint_name.lower()
        query = f"SELECT MAX(dt) as max_dt FROM {table_name} WHERE {col_name} IS NOT NULL"
    else:
        if "symbol" in params:
            symbol = params["symbol"]
            query = f"SELECT MAX(dt) as max_dt FROM {table_name} WHERE symbol = '{symbol}'"
        else:
            return None

    should_close = conn is None
    try:
        df = conn.execute(query).df()
        if not df.empty and df["max_dt"].iloc[0] is not None:
            return pd.to_datetime(df["max_dt"].iloc[0])
        return None
    except Exception:
        return None
    finally:
        if should_close:
            conn.close()

def get_exchange_timezone(conn: DuckDBPyConnection, symbol: str) -> str:
   """
   Determine the timezone for a symbol based on its Exchange in OVERVIEW.
   Defaults to 'US/Eastern'.
   """
   try:
         # Check cache/DB for Overview
         res = conn.execute(f"SELECT Exchange, Currency FROM OVERVIEW WHERE Symbol = '{symbol}'").df()
         if not res.empty:
            exchange = res.iloc[0]["Exchange"]
            
            # Load mappings from settings
            tz_map = settings.get("exchange_timezones", {})
            for tz, exchanges in tz_map.items():
               if exchange in exchanges:
                     return tz
                     
         return "US/Eastern" # Default
   except Exception:
         return "US/Eastern"

def get_numeric_columns(table_name: str) -> list[str]:
    """
    Parses the SQL schema for the given table and returns a list of column names
    that should be treated as numeric (DECIMAL, INT, BIGINT, FLOAT).
    """
    if table_name not in avs.TABLE_SCHEMAS:
        return []

    schema_sql = avs.TABLE_SCHEMAS[table_name]
    numeric_cols = []
    
    # Simple parsing of the CREATE TABLE statement
    # We look for lines like "column_name TYPE,"
    for line in schema_sql.splitlines():
        line = line.strip()
        if not line or line.startswith("CREATE TABLE") or line.startswith(");"):
            continue
            
        # Remove trailing comma and comments
        line = line.split(",")[0].split("--")[0].strip()
        
        parts = line.split()
        if len(parts) >= 2:
            col_name = parts[0].strip('"') # Handle quoted identifiers
            col_type = parts[1].upper()
            
            if any(x in col_type for x in ["DECIMAL", "INT", "BIGINT", "FLOAT", "DOUBLE", "REAL", "NUMERIC"]):
                numeric_cols.append(col_name)
                
    return numeric_cols

def get_from_db(conn: DuckDBPyConnection, endpoint_name: str, params: Dict) -> pd.DataFrame:
   """
   Attempts to retrieve data from the database.
   """
   # Determine table name
   table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper() # Default to self-named table
   
   if table_name == "MACRO":
      col_name = endpoint_name.lower()
      query = f"SELECT dt, {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY dt DESC"
   else:
      if "symbol" in params:
            symbol = params["symbol"]
            query = f"SELECT * FROM {table_name} WHERE symbol = '{symbol}'"
            if "date" in params:
               query += f" AND CAST(dt AS DATE) = '{params['date']}'"
            query += " ORDER BY dt DESC"
      else:
            # Default fallback (might be too broad, but fits current schema usage)
            query = f"SELECT * FROM {table_name}"
            if "date" in params:
               query += f" WHERE CAST(dt AS DATE) = '{params['date']}'"
            query += " ORDER BY dt DESC"

   should_close = conn is None

   try:
      # Check if table exists first prevents some error logs
      # but simpler to just try/except execution
      df = conn.execute(query).df()
      if not df.empty:
            # Restore index
            if "dt" in df.columns:
               df = df.assign(dt=pd.to_datetime(df["dt"]))
               df.set_index("dt", inplace=True)

      return df
   except Exception:
      # Likely table doesn't exist or column missing
      return pd.DataFrame()
      if should_close:
            conn.close()

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

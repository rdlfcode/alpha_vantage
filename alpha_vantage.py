"""
Alpha Vantage Client

A comprehensive client for interacting with the Alpha Vantage API.
Handles data fetching, parsing, caching, and rate limiting in a unified interface.
"""

import os
import time
import requests
import pandas as pd
import duckdb
from duckdb import DuckDBPyConnection
import logging
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, List, Union
from dotenv import load_dotenv
from alpha_vantage_schema import (
    BASE_URL,
    SYMBOL_ENDPOINTS,
    MACRO_ENDPOINTS,
    TABLE_SCHEMAS,
)
from utils import read_stock_symbols, get_endpoints
from rate_limiter import RateLimiter
from settings import settings

# Load environment variables
load_dotenv()

# =============================================================================
# LOGGING SETUP
# =============================================================================

log_settings = settings.get("logging", {})
logging.basicConfig(**log_settings)
logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    A comprehensive client for interacting with the Alpha Vantage API.

    Handles data fetching, parsing, caching, and rate limiting in a unified interface.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: Optional[str] = None,
        db_conn: Optional[Union[DuckDBPyConnection, str]] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
    ):
        """
        Initialize the Alpha Vantage client.

        Args:
           api_key: Alpha Vantage API key. If None, will try to get from environment.
           data_dir: Directory for caching data files
           db_conn: Optional DuckDB connection object or path to SQLite database for data storage
           requests_per_minute: Rate limit for requests per minute
           requests_per_day: Rate limit for requests per day
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter."
            )

        # Use settings with fallbacks
        rpm = requests_per_minute or settings.get("AlphaVantageRPM", 75)
        rpd = requests_per_day or settings.get("AlphaVantageRPD", 25)
        self.rate_limiter = RateLimiter(requests_per_minute=rpm, requests_per_day=rpd)
        self.logger = logger

        # Ensure data directory exists
        self.data_dir = Path(data_dir or settings.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(db_conn, DuckDBPyConnection):
            self.conn = db_conn
        elif isinstance(db_conn, str):
            self.conn = duckdb.connect(db_conn)
        else:
            self.conn = duckdb.connect(settings.get("db_path", "data/alpha_vantage.db"))

    def _generate_filepath(self, endpoint_name: str, params: Dict) -> Path:
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
        path = self.data_dir / "files" / endpoint_name / filename

        return path

    def _fetch_data(
        self, endpoint_name: str, params: Dict
    ) -> Optional[Union[Dict, str, None]]:
        """
        Fetches data from a single Alpha Vantage endpoint.

        Args:
           endpoint_name: The API endpoint function name
           params: Parameters for the API call

        Returns:
           Raw API response as dict (for JSON) or string (for CSV), or None if error
        """
        self.rate_limiter.wait_if_needed()

        full_params = {
            "function": endpoint_name,
            "apikey": self.api_key,
            **params,
        }

        self.logger.info(f"Fetching data for {endpoint_name} with params: {params}")

        try:
            response = requests.get(BASE_URL, params=full_params, timeout=30)
            response.raise_for_status()
            response_text = response.text

            # Check for specific rate limit warning (daily)
            # User reported message: "standard API rate limit is 25 requests per day"
            if "standard API rate limit" in response_text and "requests per day" in response_text:
                self.logger.warning("Daily API rate limit reached (detected from API response).")
                self.rate_limiter.set_daily_limit_reached()
                return None

            # Check for API errors or Notes in the response
            # Some endpoints return JSON with "Information" or "Note" even if we asked for CSV
            if "Information" in response_text or "Note" in response_text:
                # Double check if it's the daily limit inside a JSON structure
                if "standard API rate limit" in response_text and "requests per day" in response_text:
                     self.logger.warning("Daily API rate limit reached (detected from API response).")
                     self.rate_limiter.set_daily_limit_reached()
                     return None
                
                # Check for Minute/RPM limit hint
                # Message: "Please consider spreading out your free API requests more sparingly."
                if "consider spreading out your free API requests" in response_text:
                    self.logger.warning("API rate limit hint received (RPM). Sleeping 60s and retrying.")
                    time.sleep(60)
                    return self._fetch_data(endpoint_name, params)

                self.logger.warning(
                    f"API returned an error or note for {endpoint_name}: {response_text[:200]}..."
                )
                return None

            # Return appropriate format
            if params.get("datatype", "json") == "json":
                return response.json()
            else:
                return response.text

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data for {endpoint_name}: {e}")
            return None

    def _parse_response(
        self, endpoint_name: str, data: Optional[Union[Dict, str]], params: Dict
    ) -> pd.DataFrame:
        """
        Parses the raw API response into a pandas DataFrame.

        Args:
           endpoint_name: The API endpoint function name
           data: Raw API response
           params: Original request parameters

        Returns:
           Parsed DataFrame
        """
        df = pd.DataFrame()

        if data is None:
            return df

        try:
            if params.get("datatype", "json") == "csv" and isinstance(data, str):
                # Parse CSV response
                df = pd.read_csv(StringIO(data), na_values=["."])

            elif isinstance(data, dict):
                # Parse JSON response
                if any("Time Series" in k for k in data.keys()):
                    # Time series data
                    time_series_key = next(k for k in data.keys() if "Time Series" in k)
                    time_series_data = data[time_series_key]
                    if isinstance(time_series_data, dict):
                        # Convert dict of dicts to list of dicts with 'dt' column
                        records = []
                        for date_str, values in time_series_data.items():
                            if isinstance(values, dict):
                                record = values.copy()
                                record["dt"] = date_str
                                records.append(record)
                        
                        df = pd.DataFrame(records)
                        
                        if "dt" in df.columns:
                            df["dt"] = pd.to_datetime(df["dt"])
                            df.set_index("dt", inplace=True)
                            df.index.name = "dt"
                        
                        df = df.apply(pd.to_numeric, errors="coerce")
                        
                        # Rename columns to remove "1. ", "2. ", etc.
                        df.rename(
                            columns=lambda c: c.split(". ")[1] if ". " in c else c,
                            inplace=True,
                        )

                elif any("data" in k.lower() for k in data.keys()):
                    # Economic indicators and commodities
                    data_key = next(k for k in data.keys() if "data" in k.lower())
                    time_series_data = data[data_key]
                    if isinstance(time_series_data, list):
                        df = pd.DataFrame(time_series_data)
                        # Convert numeric columns
                        numeric_cols = df.select_dtypes(include=["object"]).columns
                        for col in numeric_cols:
                            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

                elif any(k in data for k in ["quarterlyReports", "annualReports"]):
                    # Financial statements
                    reports_key = (
                        "quarterlyReports"
                        if "quarterlyReports" in data
                        else "annualReports"
                    )
                    df = pd.DataFrame(data[reports_key])

                else:
                    # Fallback for other structures
                    df = pd.DataFrame([data])

            # Standardize datetime column
            for col_name in [
                "timestamp",
                "fiscalDateEnding",
                "date",
                "transactionDate",
            ]:
                if col_name in df.columns:
                    df.rename(columns={col_name: "dt"}, inplace=True)
                    break

            if "dt" in df.columns:
                df.loc[:, "dt"] = pd.to_datetime(df["dt"])
                df.set_index("dt", inplace=True)
            elif df.index.name in ["timestamp", "fiscalDateEnding", "date"]:
                df.index.name = "dt"

            if endpoint_name in SYMBOL_ENDPOINTS and "symbol" in params.keys():
                df.loc[:, "symbol"] = params["symbol"]

        except Exception as e:
            self.logger.error(f"Error parsing data for {endpoint_name}: {e}")
            return pd.DataFrame()

        return df
        
    def _get_from_db(self, endpoint_name: str, params: Dict) -> pd.DataFrame:
        """
        Attempts to retrieve data from the database.
        """
        # Determine table name
        if endpoint_name in MACRO_ENDPOINTS:
            table_name = "MACRO"
            col_name = endpoint_name.lower()
            query = f"SELECT dt, {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY dt DESC"
        else:
            table_name = endpoint_name.upper()
            if "symbol" in params:
                symbol = params["symbol"]
                query = f"SELECT * FROM {table_name} WHERE symbol = '{symbol}' ORDER BY dt DESC"
            else:
                # Default fallback (might be too broad, but fits current schema usage)
                query = f"SELECT * FROM {table_name} ORDER BY dt DESC"

        should_close = self.conn is None

        try:
            # Check if table exists first prevents some error logs
            # but simpler to just try/except execution
            df = self.conn.execute(query).df()
            if not df.empty:
                # Restore index
                if "dt" in df.columns:
                    df["dt"] = pd.to_datetime(df["dt"])
                    df.set_index("dt", inplace=True)

            return df
        except Exception:
            # Likely table doesn't exist or column missing
            return pd.DataFrame()
        finally:
            if should_close:
                self.conn.close()

    def fetch_and_cache_data(self, endpoint_name: str, params: Dict, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a single endpoint call.
        Checks DB first, then falls back to API.
        """
        if not force_refresh:
            # 1. Try DB first if not forcing refresh
            df = self._get_from_db(endpoint_name, params)
            if not df.empty:
                self.logger.info(f"Loaded {len(df)} rows from database for {endpoint_name}")
                return df

            # 1.5 Try Parquet Cache (Fallback if DB miss)
            filepath = self._generate_filepath(endpoint_name, params)
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath, engine='pyarrow')
                    self.logger.info(f"Loading from cache: {filepath}")
                    # Ensure index is set if it was reset
                    if "dt" in df.columns:
                        df["dt"] = pd.to_datetime(df["dt"])
                        df.set_index("dt", inplace=True)
                    return df
                except Exception as e:
                    self.logger.error(f"Error reading parquet cache: {e}")

        # 2. Fetch fresh data (API)
        raw_data = self._fetch_data(endpoint_name, params)
        df = self._parse_response(endpoint_name, raw_data, params)

        if df.empty:
            return df

        # 3. Save to Parquet (Redundancy)
        # Generate cache filename
        filepath = self._generate_filepath(endpoint_name, params)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, engine='pyarrow')
            self.logger.info(f"Saved to cache: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving parquet cache: {e}")

        # 4. Save to Database
        self._save_to_database(endpoint_name, params, df)

        return df

    def _save_to_database(self, endpoint_name: str, params: Dict, df: pd.DataFrame, conn: Optional[DuckDBPyConnection] = None):
        """
        Save DataFrame to SQLite database using the correct schema for each endpoint.
        """
        if df.empty:
            self.logger.warning(f"No data to save for {endpoint_name} with params {params}")
            return

        conn = conn or self.conn
        should_close = False

        try:
            # Prepare dataframe for insertion
            df_to_save = df.reset_index() if df.index.name == "dt" else df.copy()

            if endpoint_name in MACRO_ENDPOINTS:
                table_name = "MACRO"
                col_name = endpoint_name.lower()

                # Check if we need to rename the value column
                # Typically macro responses have 'value' or similar.
                # Let's inspect columns. If specific col_name not present, assume the first numeric/value column is it.
                if col_name not in df_to_save.columns:
                  # Find a suitable column to map
                  candidates = [
                     c for c in df_to_save.columns if c not in ["dt", "date"]
                  ]
                  if candidates:
                     df_to_save.rename(
                           columns={candidates[0]: col_name}, inplace=True
                     )

                # We only want to insert dt and the specific value column
                # This might result in separate rows for separate indicators given the simple Schema
                if col_name in df_to_save.columns and "dt" in df_to_save.columns:
                    # Construct a specialized insert
                    # Since we can't easily do partial column insert via "BY NAME" if other columns are missing from DF
                    # (DuckDB BY NAME expects matching columns, might error on missing ones if not nullable? No, standard SQL allows missing cols if nullable)
                    # But 'BY NAME' uses the dataframe columns.
                    # So we filter df_to_save to just [dt, col_name]
                    df_subset = df_to_save[["dt", col_name]]
                    conn.execute(
                        f"INSERT INTO {table_name} BY NAME SELECT * FROM df_subset"
                    )
                else:
                    self.logger.warning(f"Could not map columns for MACRO table insert: {df_to_save.columns}")
                    return
            else:
                table_name = endpoint_name.upper()
                conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save")

            self.logger.info(f"Saved {len(df)} rows to {table_name}")

        except Exception as e:
            # Handle missing table
            if isinstance(e, duckdb.CatalogException) or "does not exist" in str(e):
                try:
                    # Create table
                    schema_sql = TABLE_SCHEMAS.get(endpoint_name)
                    if endpoint_name in MACRO_ENDPOINTS:
                        schema_sql = TABLE_SCHEMAS.get("MACRO")

                    if schema_sql:
                        conn.execute(schema_sql)
                        # Retry insert (recursive call? or just copy logic)
                        # Let's just retry the execute logic
                        if endpoint_name in MACRO_ENDPOINTS:
                            col_name = endpoint_name.lower()
                            if col_name in df_to_save.columns:
                                df_subset = df_to_save[["dt", col_name]]
                                conn.execute(
                                    f"INSERT INTO {table_name} BY NAME SELECT * FROM df_subset"
                                )
                        else:
                            conn.execute(
                                f"INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save"
                            )

                        self.logger.info(
                            f"Created table and saved {len(df)} rows to {table_name}"
                        )
                    else:
                        self.logger.error(f"No schema found for {endpoint_name}")
                except Exception as create_error:
                    self.logger.error(
                        f"Error creating table/saving for {endpoint_name}: {create_error}"
                    )
            else:
                self.logger.error(f"Error saving to database for {endpoint_name}: {e}")

        finally:
            if should_close:
                conn.close()

    def get_data(
        self,
        symbols: Optional[List[str]] = None,
        endpoints: Optional[Dict[str, Dict]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for given symbols and endpoints.
        Gets all data by default.

        Args:
           symbols: List of stock symbols.
           endpoints: Dictionary of endpoint configurations.
           start_date: Start date for filtering (YYYY-MM-DD).
           end_date: End date for filtering (YYYY-MM-DD).
           force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
           A combined DataFrame with all requested data.
        """
        try:
            symbols = symbols or read_stock_symbols()

            endpoints = endpoints or get_endpoints()

            all_dfs = []

            # Process symbol-specific endpoints
            symbol_endpoints = {
                k: v for k, v in endpoints.items() if k in SYMBOL_ENDPOINTS
            }
            for symbol in symbols:
                for endpoint_name, params in symbol_endpoints.items():
                    try:
                        # Create a copy of params to avoid modifying the original dict in the loop
                        current_params = params.copy()
                        current_params.update({"symbol": symbol})
                        df = self.fetch_and_cache_data(
                            endpoint_name, current_params, force_refresh
                        )
                        if not df.empty:
                            all_dfs.append(df)
                    except Exception as e:
                        self.logger.error(
                            f"Error fetching {endpoint_name} for {symbol}: {e}"
                        )

            # Process macro-economic endpoints
            macro_endpoints_map = {
                k: v for k, v in endpoints.items() if k in MACRO_ENDPOINTS
            }
            for endpoint_name, params in macro_endpoints_map.items():
                try:
                    df = self.fetch_and_cache_data(
                        endpoint_name, params, force_refresh
                    )
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    self.logger.error(
                        f"Error fetching macro endpoint {endpoint_name}: {e}"
                    )

            if not all_dfs:
                self.logger.warning(
                    "No data fetched for the given symbols and endpoints."
                )
                return pd.DataFrame()

            # Combine all DataFrames
            combined_df = pd.concat(all_dfs, sort=False)

            # Filter by date range if specified
            if "date" in combined_df.index.names:
                if start_date:
                    combined_df = combined_df[combined_df.index >= start_date]
                if end_date:
                    combined_df = combined_df[combined_df.index <= end_date]

            self.logger.info(
                f"Successfully built dataset with {len(combined_df)} rows."
            )
            return combined_df

        except Exception as e:
            self.logger.error(f"Error in get_data: {e}", exc_info=True)
            return pd.DataFrame()

    def get_dataset_from_db(
        self, sql_query: str, conn: Optional[DuckDBPyConnection] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query against the database and return results as DataFrame.

        Args:
           sql_query: SQL query to execute

        Returns:
           Query results as DataFrame
        """
        try:
            conn = conn or duckdb.connect(self.db_path)
            df = conn.execute(sql_query).df()
            conn.close()
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return pd.DataFrame()

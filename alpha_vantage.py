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
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from utils import read_stock_symbols, get_endpoints
from rate_limiter import RateLimiter
from settings import settings
from alpha_vantage_schema import (
    BASE_URL,
    SYMBOL_ENDPOINTS,
    MACRO_ENDPOINTS,
    FUNDAMENTAL_ENDPOINTS,
    TABLE_SCHEMAS,
    ENDPOINT_TO_TABLE_MAP,
)

# Load environment variables
load_dotenv()

# =============================================================================
# LOGGING SETUP
# =============================================================================
log_settings = settings.get("logging", {})
# Remove filename from basicConfig if we want custom handlers effectively mixed?
# Actually basicConfig is fine, but we might want to ensure TQDM plays nice.
# For now, let's keep basicConfig but add a specific handler class.

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# Only add TqdmHandler if not already there to avoid dupes on re-run
has_tqdm_handler = any(isinstance(h, TqdmLoggingHandler) for h in logging.getLogger().handlers)
if not has_tqdm_handler:
    logging.getLogger().addHandler(TqdmLoggingHandler())

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
            
            # Try to parse as JSON first to check for API errors (which are always JSON)
            try:
                data = response.json()
                is_json = True
            except ValueError:
                data = response.text
                is_json = False

            # Check for API errors/notes in JSON structure
            if is_json and isinstance(data, dict):
                # Check for rate limit specific messages
                # Messages can be in "Information" or "Note"
                msg_val = data.get("Information", "") or data.get("Note", "") or data.get("Error Message", "")
                
                if msg_val:
                    # Check for Daily limit
                    if "standard API rate limit" in str(msg_val) and "requests per day" in str(msg_val):
                        self.logger.warning("Daily API rate limit reached (detected from API response).")
                        self.rate_limiter.set_daily_limit_reached()
                        return None
                    
                    # Check for RPM limit hint
                    if "consider spreading out your free API requests" in str(msg_val) or "Burst pattern detected" in str(msg_val):
                        self.logger.warning("API rate limit hint received (RPM). Sleeping 60s and retrying.")
                        time.sleep(60)
                        return self._fetch_data(endpoint_name, params)

                    # Any other Information/Note/Error at top level is likely an error or blocking note
                    # (Meta Data 'Information' is NOT at top level, so this is safe)
                    self.logger.warning(
                        f"API returned an error or note for {endpoint_name}: {msg_val}"
                    )
                    return None

            # Return appropriate format
            requested_format = params.get("datatype", "json")
            if requested_format == "json":
                if is_json:
                    return data
                else:
                    self.logger.warning(f"Expected JSON but got non-JSON response for {endpoint_name}")
                    return None
            else:
                # requested CSV
                if is_json:
                     # If we executed the error check above and passed, it means we got a JSON that isn't an error?
                     # But we wanted CSV. Alpha Vantage returns JSON for errors, but maybe empty JSON or weird JSON if verified above?
                     # Usually if we want CSV, we shouldn't get JSON unless it Is an error.
                     # But we already checked for error keys.
                     # It might be that the API returned a JSON we can parse (like HISTORICAL_OPTIONS sometimes does)
                     # or it's a valid JSON response despite us asking for CSV.
                     # If it looks like valid data (not an error), let's return it.
                     self.logger.warning(f"Expected CSV but got JSON response for {endpoint_name}. Attempting to parse as JSON.")
                     return data
                return data

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
                            df = df.assign(dt=pd.to_datetime(df["dt"]))
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


                elif endpoint_name in FUNDAMENTAL_ENDPOINTS:
                    # Fundamental Data (Income Statement, Balance Sheet, Cash Flow, Earnings)
                    records = []
                    symbol = params.get("symbol")
                    
                    # Mapping of report types keys to standardized keys
                    report_map = {
                        "annualReports": ("ANNUAL", endpoint_name),
                        "quarterlyReports": ("QUARTERLY", endpoint_name),
                        "annualEarnings": ("ANNUAL", "EARNINGS"),
                        "quarterlyEarnings": ("QUARTERLY", "EARNINGS")
                    }

                    for report_key, (period_type, report_type) in report_map.items():
                         if report_key in data:
                             report_list = data[report_key]
                             if isinstance(report_list, list):
                                 for item in report_list:
                                     # Determine Date
                                     dt_val = item.get("fiscalDateEnding") or item.get("reportedDate")
                                     if not dt_val:
                                         continue

                                     # Iterate metrics
                                     for key, val in item.items():
                                         if key in ["fiscalDateEnding", "reportedDate", "reportedCurrency", "symbol"]:
                                             # Metadata columns, skip for value list (or handle reportedCurrency separately if needed)
                                             continue
                                         
                                         # Convert value to numeric
                                         try:
                                             if val is None or val == "None":
                                                 numeric_val = None
                                             else:
                                                 numeric_val = float(val)
                                         except (ValueError, TypeError):
                                             numeric_val = None
                                             
                                         if numeric_val is not None:
                                             records.append({
                                                 "symbol": symbol,
                                                 "dt": dt_val,
                                                 "period_type": period_type,
                                                 "report_type": report_type,
                                                 "metric": key,
                                                 "value": numeric_val
                                             })
                    
                    df = pd.DataFrame(records)
                    
                else:
                    # Fallback for other structures
                    df = pd.DataFrame([data])

            # Standardize datetime column
            for col_name in [
                "timestamp",
                "fiscalDateEnding",
                "date",
                "transactionDate",
                "transaction_date",
            ]:
                if col_name in df.columns:
                    df.rename(columns={col_name: "dt"}, inplace=True)
                    break 
            
            # Standardize symbol column
            if "ticker" in df.columns and "symbol" not in df.columns:
                df.rename(columns={"ticker": "symbol"}, inplace=True)

            # Additional renames for Insider Transactions
            rename_map = {
                "executive": "reportingPerson",
                "acquisition_or_disposal": "transactionType",
                "share_price": "price"
            }
            # Only rename if columns exist
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

            if "dt" in df.columns:
                df = df.assign(dt=pd.to_datetime(df["dt"]))
                df.set_index("dt", inplace=True)
            elif df.index.name in ["timestamp", "fiscalDateEnding", "date"]:
                df.index.name = "dt"

            if endpoint_name in SYMBOL_ENDPOINTS and "symbol" in params.keys():
                if not df.empty:
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
        table_name = ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper() # Default to self-named table
        
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

        should_close = self.conn is None

        try:
            # Check if table exists first prevents some error logs
            # but simpler to just try/except execution
            df = self.conn.execute(query).df()
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
                self.conn.close()

    def _get_latest_date_from_db(self, endpoint_name: str, params: Dict) -> Optional[datetime]:
        """
        Gets the latest date available for a given endpoint/symbol from the DB.
        """
        table_name = ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()

        if table_name == "MACRO":
            col_name = endpoint_name.lower()
            query = f"SELECT MAX(dt) as max_dt FROM {table_name} WHERE {col_name} IS NOT NULL"
        else:
            if "symbol" in params:
                symbol = params["symbol"]
                query = f"SELECT MAX(dt) as max_dt FROM {table_name} WHERE symbol = '{symbol}'"
            else:
                return None

        should_close = self.conn is None
        try:
            df = self.conn.execute(query).df()
            if not df.empty and df["max_dt"].iloc[0] is not None:
                return pd.to_datetime(df["max_dt"].iloc[0])
            return None
        except Exception:
            return None
        finally:
            if should_close:
                self.conn.close()

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

            # Identify target table name
            table_name = ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()

            if table_name == "MACRO":
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
                    
                    # Delete overlaps first
                    if not df_subset.empty:
                        min_dt = df_subset["dt"].min()
                        max_dt = df_subset["dt"].max()
                        conn.execute(
                            f"DELETE FROM {table_name} WHERE {col_name} IS NOT NULL AND dt >= '{min_dt}' AND dt <= '{max_dt}'"
                        )
                        
                    conn.execute(
                        f"INSERT INTO {table_name} BY NAME SELECT * FROM df_subset"
                    )
                else:
                    self.logger.warning(f"Could not map columns for MACRO table insert: {df_to_save.columns}")
                    return
            else:
                # Delete overlaps first
                if "dt" in df_to_save.columns:
                    min_dt = df_to_save["dt"].min()
                    max_dt = df_to_save["dt"].max()
                    
                    # If duplicate symbols exist in a generic table, refine delete by symbol
                    symbol_clause = ""
                    if "symbol" in df_to_save.columns:
                        symbol_val = df_to_save["symbol"].iloc[0] # Assume same symbol per batch usually
                        symbol_clause = f"AND symbol = '{symbol_val}'"
                    
                    # Special handling for FUNDAMENTALS: only delete for specific report_type
                    report_type_clause = ""
                    if table_name == "FUNDAMENTALS" and "report_type" in df_to_save.columns:
                        # Assume one report type per save batch (safe for single endpoint call)
                        report_type_val = df_to_save["report_type"].iloc[0]
                        report_type_clause = f"AND report_type = '{report_type_val}'"

                    conn.execute(
                        f"DELETE FROM {table_name} WHERE dt >= '{min_dt}' AND dt <= '{max_dt}' {symbol_clause} {report_type_clause}"
                    )
                
                # Filter columns to match table schema to avoid Binder Error on extra columns
                try:
                    db_cols_df = conn.execute(f"PRAGMA table_info('{table_name}')").df()
                    if not db_cols_df.empty:
                        valid_cols_set = set(db_cols_df['name'].tolist())
                        
                        # Deduplicate DF columns
                        df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
                        
                        # Filter to valid cols
                        existing_cols = [c for c in df_to_save.columns if c in valid_cols_set]
                        if existing_cols:
                            df_to_save = df_to_save[existing_cols]
                        else:
                            self.logger.warning(f"No matching columns for {table_name} after filtering. Skipping.")
                            return
                except Exception as filter_err:
                    self.logger.warning(f"Error filtering columns for {table_name}: {filter_err}")
                
                conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save")

            self.logger.info(f"Saved {len(df)} rows to {table_name}")

        except Exception as e:
            # Handle missing table
            if isinstance(e, duckdb.CatalogException) or "does not exist" in str(e):
                try:
                    # Create table
                    # Use table name to look up schema if key matches, OR endpoint name?
                    # The schemas are keyed by ENDPOINT usually, except MACRO.
                    # Mappings: 
                    # MACRO -> MACRO
                    # TIME_SERIES_DAILY -> TIME_SERIES_DAILY
                    # HISTORICAL_OPTIONS -> HISTORICAL_OPTIONS
                    
                    # If table_name is MACRO, we use MACRO schema.
                    # Otherwise, table definition usually matches the table name (which is endpoint name).
                    
                    schema_key = table_name if table_name in ["MACRO", "FUNDAMENTALS"] else endpoint_name
                    schema_sql = TABLE_SCHEMAS.get(schema_key)
                    
                    if not schema_sql and table_name == "MACRO": 
                        schema_sql = TABLE_SCHEMAS.get("MACRO") # fallback

                    if schema_sql:
                        conn.execute(schema_sql)
                        # Retry insert (recursive call? or just copy logic)
                        # Let's just retry the execute logic
                        if table_name == "MACRO":
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

    def fetch_and_cache_data(self, endpoint_name: str, params: Dict, force_refresh: bool = False, min_required_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a single endpoint call.
        Checks DB first, then falls back to API.
        """
        if not force_refresh:
            # 1. Try DB first if not forcing refresh
            df = self._get_from_db(endpoint_name, params)
            if not df.empty:
                # Check if data is fresh enough (if min_required_date is set)
                if min_required_date:
                    latest_dt = df.index.max()
                    if latest_dt < min_required_date:
                        self.logger.info(f"Data for {endpoint_name} is stale (latest: {latest_dt}, required: {min_required_date}). Fetching fresh...")
                        df = pd.DataFrame() # Trigger fetch
                    else:
                        self.logger.info(f"Loaded {len(df)} rows from database for {endpoint_name}")
                        return df
                else:
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

                    if min_required_date and not df.empty:
                        latest_dt = None
                        if isinstance(df.index, pd.DatetimeIndex):
                            latest_dt = df.index.max()
                        elif "dt" in df.columns:
                            latest_dt = pd.to_datetime(df["dt"]).max()
                        
                        if latest_dt and latest_dt < min_required_date:
                            self.logger.info(f"Cached data for {endpoint_name} is stale (latest: {latest_dt}, required: {min_required_date}). Fetching fresh...")
                            # Don't return, let it fall through to API fetch
                        else:
                            return df
                    else:
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
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            start_date = start_date or (pd.to_datetime(end_date) - pd.DateOffset(years=15)).strftime("%Y-%m-%d")

            # --- Build Task List ---
            tasks = []

            # 1. Symbol-specific tasks
            symbol_endpoints = {
                k: v for k, v in endpoints.items() if k in SYMBOL_ENDPOINTS
            }
            for symbol in symbols:
                for endpoint_name, params in symbol_endpoints.items():
                    # Create a copy of params to avoid modifying the original dict
                    task_params = params.copy()
                    task_params["symbol"] = symbol
                    
                    if "date" in params.keys():
                        # Determine date range
                        s_date = pd.to_datetime(start_date)
                        e_date = pd.to_datetime(end_date)
                        
                        # Get existing dates from DB to avoid redundant fetches
                        try:
                            table_name = ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()
                            # Efficiently query just the dates
                            existing_df = self.conn.execute(f"SELECT DISTINCT dt FROM {table_name} WHERE symbol = '{symbol}' ORDER BY dt").df()
                            if not existing_df.empty:
                                existing_dates = pd.to_datetime(existing_df['dt']).dt.normalize()
                            else:
                                existing_dates = pd.Index([])
                        except Exception:
                            # Table might not exist yet
                            existing_dates = pd.Index([])

                        full_range = pd.date_range(start=s_date, end=e_date)
                        needed_dates = full_range.difference(existing_dates)
                        
                        for date in needed_dates:
                            p = task_params.copy()
                            p["date"] = date.strftime("%Y-%m-%d")
                            tasks.append((endpoint_name, p))
                    else:
                        tasks.append((endpoint_name, task_params))

            # 2. Macro-economic tasks
            macro_endpoints_map = {
                k: v for k, v in endpoints.items() if k in MACRO_ENDPOINTS
            }
            for endpoint_name, params in macro_endpoints_map.items():
                tasks.append((endpoint_name, params))

            return self._execute_tasks(tasks, force_refresh, start_date, end_date)

        except Exception as e:
            self.logger.error(f"Error in get_data: {e}", exc_info=True)
            return pd.DataFrame()

    def _execute_tasks(
        self, 
        tasks: List, 
        force_refresh: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Internal helper to execute a list of tasks concurrently with tqdm.
        """
        if not tasks:
            return pd.DataFrame()

        max_workers = settings.get("MAX_CONCURRENT_REQUESTS", 5)
        all_dfs = []
        
        self.logger.info(f"Starting execution of {len(tasks)} tasks with {max_workers} workers.")

        min_required_datetime = pd.to_datetime(end_date) if end_date else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures to tasks
            future_to_task = {
                executor.submit(
                    self.fetch_and_cache_data, 
                    endpoint_name, 
                    params, 
                    force_refresh,
                    min_required_datetime
                ): (endpoint_name, params)
                for endpoint_name, params in tasks
            }

            # Progress bar
            with tqdm(total=len(tasks), unit="req", desc="Fetching Data") as pbar:
                for future in as_completed(future_to_task):
                    endpoint_name, params = future_to_task[future]
                    try:
                        df = future.result()
                        if not df.empty and len(df.dropna()) > 0:
                            all_dfs.append(df)
                        
                        pbar.update(1)
                        symbol_info = f" ({params.get('symbol')})" if 'symbol' in params else ""
                        pbar.set_postfix_str(f"Last: {endpoint_name}{symbol_info}", refresh=False)

                    except Exception as e:
                        self.logger.error(f"Task failed for {endpoint_name}: {e}")
                        pbar.update(1)

        if not all_dfs:
            self.logger.warning("No data returned from any tasks.")
            return pd.DataFrame()

        # Combine all DataFrames
        self.logger.info("Concatenating results...")
        combined_df = pd.concat(all_dfs, sort=False)

        # Filter by date range if specified
        if "dt" in combined_df.index.names:
            if start_date:
                combined_df = combined_df[combined_df.index >= start_date]
            if end_date:
                combined_df = combined_df[combined_df.index <= end_date]
        
        return combined_df

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

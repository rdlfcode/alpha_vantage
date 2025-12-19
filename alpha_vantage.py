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

import utils
from rate_limiter import RateLimiter
from settings import settings
import alpha_vantage_schema as avs

# Load environment variables
load_dotenv()

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
           api_key: Alpha Vantage API key.
           data_dir: Directory for caching data files.
           db_conn: DuckDB connection or path to database.
           requests_per_minute: Rate limit per minute.
           requests_per_day: Rate limit per day.
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter."
            )

        rpm = requests_per_minute or settings.get("AlphaVantageRPM", 75)
        rpd = requests_per_day or settings.get("AlphaVantageRPD", 25)
        self.rate_limiter = RateLimiter(requests_per_minute=rpm, requests_per_day=rpd)
        self.logger = logger

        self.data_dir = Path(data_dir or settings.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(db_conn, DuckDBPyConnection):
            self.conn = db_conn
        elif isinstance(db_conn, str):
            self.conn = duckdb.connect(db_conn)
        else:
            self.conn = duckdb.connect(settings.get("db_path", "data/alpha_vantage.db"))

    def _fetch_data(self, endpoint_name: str, params: Dict) -> Optional[Union[Dict, str, None]]:
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
            response = requests.get(avs.BASE_URL, params=full_params, timeout=30)
            response.raise_for_status()
            
            # Try to parse as JSON first to check for API errors (which are always JSON)
            try:
                data = response.json()
                is_json = True
            except ValueError:
                data = response.text
                is_json = False

            if is_json and isinstance(data, dict):
                msg_val = data.get("Information", "") or data.get("Note", "") or data.get("Error Message", "")
                
                if msg_val:
                    if "standard API rate limit" in str(msg_val) and "requests per day" in str(msg_val):
                        self.logger.warning("Daily API rate limit reached.")
                        self.rate_limiter.set_daily_limit_reached()
                        return None
                    
                    if "consider spreading out your free API requests" in str(msg_val) or "Burst pattern detected" in str(msg_val):
                        self.logger.warning("API rate limit hint received (RPM). Sleeping 60s.")
                        time.sleep(60)
                        return self._fetch_data(endpoint_name, params)

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

    def _parse_response(self, endpoint_name: str, data: Optional[Union[Dict, str]], params: Dict) -> pd.DataFrame:
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


                elif endpoint_name in avs.FUNDAMENTAL_ENDPOINTS:
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
                    # Fallback for other structures (e.g. OVERVIEW)
                    df = pd.DataFrame([data])
                    
                    # Clean "None", "-", "0000-00-00" before standardizing
                    df.replace(["None", "none", "-", "0000-00-00"], [None, None, None, None], inplace=True)
                    
                    # Add timestamp for point-in-time reference if missing
                    if "dt" not in df.columns:
                        df["dt"] = pd.Timestamp.now("UTC")
                        
                    if "dt" not in df.columns:
                        df["dt"] = pd.Timestamp.now("UTC")
                        
                    # Standardize numeric columns based on schema
                    # prevents "Conversion Error" in DuckDB when it expects specific types
                    table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()
                    numeric_cols = utils.get_numeric_columns(table_name)
                    
                    for col in numeric_cols:
                        if col in df.columns:
                            # Force Coerce keys that are supposed to be numeric
                            # This handles "None", "-", empty strings, etc. by turning them into NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')

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
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

            if "dt" in df.columns:
                df = df.assign(dt=pd.to_datetime(df["dt"]))
                df.set_index("dt", inplace=True)
            elif df.index.name in ["timestamp", "fiscalDateEnding", "date"]:
                df.index.name = "dt"

            if endpoint_name in avs.SYMBOL_ENDPOINTS and "symbol" in params.keys():
                if not df.empty:
                    df.loc[:, "symbol"] = params["symbol"]

        except Exception as e:
            self.logger.error(f"Error parsing data for {endpoint_name}: {e}")
            return pd.DataFrame()

        # ENSURE UTC
        if not df.empty and "dt" in df.columns:
            try:
                if df["dt"].dt.tz is None:
                    if endpoint_name == "TIME_SERIES_INTRADAY":
                         # Localize as ET then convert to UTC
                         df["dt"] = df["dt"].dt.tz_localize("US/Eastern", ambiguous="infer").dt.tz_convert("UTC")
                    else:
                         df["dt"] = df["dt"].dt.tz_localize("UTC")
                else:
                    df["dt"] = df["dt"].dt.tz_convert("UTC")
                
                # Re-set index if it was dt
                if df.index.name == "dt":
                    df = df.assign(dt=pd.to_datetime(df["dt"]))
                    df.set_index("dt", inplace=True)
                    
            except Exception as utc_err:
                 self.logger.warning(f"Could not convert dt to UTC: {utc_err}")

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
            df_to_save = df.reset_index() if df.index.name == "dt" else df.copy()
            table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()

            # 1. Ensure Table Exists
            try:
                # Check if table exists
                tbl_exists = conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                ).fetchone()[0] > 0
                
                if not tbl_exists:
                    # Create table
                    schema_sql = avs.TABLE_SCHEMAS.get(table_name)
                    if not schema_sql and table_name == "MACRO":
                        schema_sql = avs.TABLE_SCHEMAS.get("MACRO")
                    
                    if schema_sql:
                        conn.execute(schema_sql)
                    else:
                        self.logger.error(f"No schema found for {endpoint_name} (Table: {table_name})")
                        return
            except Exception as e:
                self.logger.error(f"Error checking/creating table {table_name}: {e}")
                return

            # 2. Prepare Data (Rename/Delete overlaps)
            if table_name == "MACRO":
                col_name = endpoint_name.lower()
                if col_name not in df_to_save.columns:
                    # heuristics to find value column
                    candidates = [c for c in df_to_save.columns if c not in ["dt", "date"]]
                    if candidates:
                        df_to_save.rename(columns={candidates[0]: col_name}, inplace=True)

                if col_name in df_to_save.columns and "dt" in df_to_save.columns:
                    df_subset = df_to_save[["dt", col_name]]
                    if not df_subset.empty:
                        min_dt = df_subset["dt"].min()
                        max_dt = df_subset["dt"].max()
                        conn.execute(
                            f"DELETE FROM {table_name} WHERE {col_name} IS NOT NULL AND dt >= '{min_dt}' AND dt <= '{max_dt}'"
                        )
                    
                    df_to_save = df_subset
                else:
                    self.logger.warning(f"Could not map columns for MACRO table insert: {df_to_save.columns}")
                    return
            else:
                # Delete overlaps
                if "dt" in df_to_save.columns:
                    min_dt = df_to_save["dt"].min()
                    max_dt = df_to_save["dt"].max()
                    
                    symbol_clause = ""
                    if "symbol" in df_to_save.columns:
                        symbol_val = df_to_save["symbol"].iloc[0]
                        symbol_clause = f"AND symbol = '{symbol_val}'"
                    
                    report_type_clause = ""
                    if table_name == "FUNDAMENTALS" and "report_type" in df_to_save.columns:
                        report_type_val = df_to_save["report_type"].iloc[0]
                        report_type_clause = f"AND report_type = '{report_type_val}'"

                    conn.execute(
                        f"DELETE FROM {table_name} WHERE dt >= '{min_dt}' AND dt <= '{max_dt}' {symbol_clause} {report_type_clause}"
                    )

            # 3. Filter Columns and Insert
            try:
                # Get valid columns for the table
                db_cols_df = conn.execute(f"PRAGMA table_info('{table_name}')").df()
                if not db_cols_df.empty:
                    valid_cols_set = set(db_cols_df['name'].tolist())
                    
                    # Deduplicate DF columns
                    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
                    
                    # Filter
                    existing_cols = [c for c in df_to_save.columns if c in valid_cols_set]
                    if existing_cols:
                        df_to_save = df_to_save[existing_cols]
                        conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save")
                        self.logger.info(f"Saved {len(df_to_save)} rows to {table_name}")
                    else:
                        self.logger.warning(f"No matching columns for {table_name} after filtering.")
                else:
                    self.logger.error(f"Could not get schema info for {table_name}")

            except Exception as insert_err:
                self.logger.error(f"Error inserting into {table_name}: {insert_err}")

        except Exception as e:
            self.logger.error(f"Error save flow for {endpoint_name}: {e}")

        finally:
            if should_close:
                conn.close()

    def fetch_and_cache_data(self, endpoint_name: str, params: Dict, force_refresh: bool = False, min_required_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a single endpoint call.
        Checks DB first, then falls back to Parquet cache, then API.

        Args:
            endpoint_name: API endpoint function name.
            params: Parameters for the API call.
            force_refresh: If True, skip DB/Cache and fetch from API.
            min_required_date: If cached data is older than this, fetch fresh.

        Returns:
            DataFrame containing the requested data.
        """
        if not force_refresh:
            # 1. Try DB first
            df = utils.get_from_db(self.conn, endpoint_name, params)
            if not df.empty:
                if min_required_date:
                    latest_dt = df.index.max()
                    # Ensure timezone awareness for comparison
                    if latest_dt.tz is None:
                        latest_dt = latest_dt.tz_localize("UTC")
                    else:
                        latest_dt = latest_dt.tz_convert("UTC")
                        
                    if latest_dt < min_required_date:
                        self.logger.info(f"Data for {endpoint_name} is stale (latest: {latest_dt}). Fetching fresh...")
                        df = pd.DataFrame()
                    else:
                        self.logger.info(f"Loaded {len(df)} rows from database for {endpoint_name}")
                        return df
                else:
                    self.logger.info(f"Loaded {len(df)} rows from database for {endpoint_name}")
                    return df

            # 2. Try Parquet Cache
            filepath = utils.generate_filepath(self.data_dir, endpoint_name, params)
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath, engine='pyarrow')
                    self.logger.info(f"Loading from cache: {filepath}")
                    
                    if "dt" in df.columns:
                        df = df.assign(dt=pd.to_datetime(df["dt"]))
                        df.set_index("dt", inplace=True)
                        df.index.name = "dt"

                    if min_required_date and not df.empty:
                        latest_dt = None
                        if isinstance(df.index, pd.DatetimeIndex):
                            latest_dt = df.index.max()
                        elif "dt" in df.columns:
                            latest_dt = pd.to_datetime(df["dt"]).max()
                        
                        if latest_dt:
                            if latest_dt.tz is None:
                                latest_dt = latest_dt.tz_localize("UTC")
                            else:
                                latest_dt = latest_dt.tz_convert("UTC")

                        if latest_dt and latest_dt < min_required_date:
                            self.logger.info(f"Cached data is stale (latest: {latest_dt}). Fetching fresh...")
                        else:
                            self._save_to_database(endpoint_name, params, df)
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
        filepath = utils.generate_filepath(self.data_dir, endpoint_name, params)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, engine='pyarrow')
            self.logger.info(f"Saved to cache: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving parquet cache: {e}")

        # 4. Save to Database
        self._save_to_database(endpoint_name, params, df)

        return df

    def _should_fetch(self, symbol: str, endpoint: str, params: Dict) -> bool:
        """
        Check if we should fetch data based on existing data freshness and market status.

        Args:
            symbol: Stock symbol.
            endpoint: API endpoint name.
            params: Query parameters.

        Returns:
            True if data should be fetched, False otherwise.
        """
        try:
            # 1. Get latest date from DB
            latest_dt = utils.get_latest_date_from_db(self.conn, endpoint, params)
            if not latest_dt:
                return True # No data, must fetch

            # 2. Determine comparison time (Current time in Market Timezone)
            tz_str = utils.get_exchange_timezone(self.conn, symbol)
            try:
                # We need a timezone aware current time
                # Using pandas for easy timezone handling
                now_utc = pd.Timestamp.now("UTC")
                now_market = now_utc.tz_convert(tz_str)
                
                if latest_dt.tz is None:
                    latest_dt = latest_dt.tz_localize("UTC")
                
                latest_market = latest_dt.tz_convert(tz_str)

                # Check if data is up to date (e.g. daily data fetch once per day)
                if "DAILY" in endpoint:
                    if latest_market.date() >= now_market.date():
                        self.logger.info(f"Data for {symbol} ({endpoint}) is up to date. Skipping.")
                        return False
                
            except Exception as e:
                self.logger.warning(f"Error in smart update check: {e}. Defaulting to fetch.")
                return True
                
            return True
        except Exception:
            return True

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
        
        Args:
            symbols: List of stock symbols to fetch. Defaults to reading stocks.txt.
            endpoints: Dictionary of endpoints and parameters. Defaults to default endpoints.
            start_date: Start date for time series data.
            end_date: End date for time series data.
            force_refresh: If True, force fetch from API.

        Returns:
            Combined DataFrame with all data.
        """
        try:
            symbols = symbols or utils.read_stock_symbols()
            endpoints = endpoints or utils.get_endpoints()
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            start_date = start_date or (pd.to_datetime(end_date) - pd.DateOffset(years=15)).strftime("%Y-%m-%d")

            # --- Build Task List ---
            tasks = []
            
            # 1. Symbol-specific tasks
            avs.SYMBOL_ENDPOINTS = {
                k: v for k, v in endpoints.items() if k in avs.SYMBOL_ENDPOINTS
            }
            
            for symbol in symbols:
                for endpoint_name, params in avs.SYMBOL_ENDPOINTS.items():
                    # Create a copy of params
                    task_params = params.copy()
                    task_params["symbol"] = symbol
                    
                    if not self._should_fetch(symbol, endpoint_name, task_params) and not force_refresh:
                        continue

                    if "date" in params.keys():
                        # ... (Existing Date Logic - could be optimized similarly but leaving as involves sub-dates)
                         # Determine date range
                        s_date = pd.to_datetime(start_date)
                        e_date = pd.to_datetime(end_date)
                        
                        try:
                            table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()
                            existing_df = self.conn.execute(f"SELECT DISTINCT dt FROM {table_name} WHERE symbol = '{symbol}' ORDER BY dt").df()
                            if not existing_df.empty:
                                existing_dates = pd.to_datetime(existing_df['dt']).dt.normalize() # Ensure normalize
                                if existing_dates.tz:
                                     existing_dates = existing_dates.tz_convert(None) # removing tz for comparison with simple range
                            else:
                                existing_dates = pd.Index([])
                        except Exception:
                            existing_dates = pd.Index([])

                        full_range = pd.date_range(start=s_date, end=e_date)
                        
                        # Normalize full_range
                        full_range = full_range.normalize()
                        
                        # Set operation
                        needed_dates = full_range.difference(existing_dates)
                        
                        for date in needed_dates:
                            p = task_params.copy()
                            p["date"] = date.strftime("%Y-%m-%d")
                            tasks.append((endpoint_name, p))
                    else:
                        tasks.append((endpoint_name, task_params))

            # 2. Macro-economic tasks
            avs.MACRO_ENDPOINTS_map = {
                k: v for k, v in endpoints.items() if k in avs.MACRO_ENDPOINTS
            }
            for endpoint_name, params in avs.MACRO_ENDPOINTS_map.items():
                if not self._should_fetch("MACRO", endpoint_name, params) and not force_refresh:
                    continue
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
        Internal helper to execute a list of tasks concurrently.

        Args:
            tasks: List of (endpoint_name, params) tuples.
            force_refresh: Force refresh flag.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Concatenated DataFrame.
        """
        if not tasks:
            return pd.DataFrame()

        max_workers = settings.get("MaxConcurrentRequests", 5)
        all_dfs = []
        
        self.logger.info(f"Starting execution of {len(tasks)} tasks with {max_workers} workers.")

        min_required_datetime = pd.to_datetime(end_date).tz_localize("UTC") if end_date else None

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

        # Standardize index to UTC DatetimeIndex if "dt" is the index name
        # This handles mixed naive/aware inputs from cache vs API
        if "dt" in combined_df.index.names:
            try:
                # Ensure it's datetime
                if not isinstance(combined_df.index, pd.DatetimeIndex):
                    combined_df.index = pd.to_datetime(combined_df.index, utc=True)
                
                # If it's already DatetimeIndex, ensure it is UTC
                if combined_df.index.tz is None:
                    combined_df.index = combined_df.index.tz_localize("UTC")
                else:
                    combined_df.index = combined_df.index.tz_convert("UTC")
            except Exception as e:
                self.logger.warning(f"Failed to normalize index: {e}")

            # Filter by date range if specified
            if start_date:
                # Ensure start_date is a comparable timestamp (UTC)
                start_dt = pd.to_datetime(start_date)
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize("UTC")
                else:
                    start_dt = start_dt.tz_convert("UTC")
                combined_df = combined_df[combined_df.index >= start_dt]

            if end_date:
                # Ensure end_date is a comparable timestamp (UTC)
                end_dt = pd.to_datetime(end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize("UTC")
                else:
                    end_dt = end_dt.tz_convert("UTC")
                combined_df = combined_df[combined_df.index <= end_dt]
        
        return combined_df

    def get_dataset_from_db(self, sql_query: str, conn: Optional[DuckDBPyConnection] = None) -> pd.DataFrame:
        """
        Execute SQL query against the database and return results as DataFrame.

        Args:
           sql_query: SQL query to execute.
           conn: Optional explicit connection to use.

        Returns:
           Query results as DataFrame.
        """
        should_close = False
        try:
            if conn is None:
                # Use class connection or create new one if path is unknown (fallback)
                conn = self.conn
            
            df = conn.execute(sql_query).df()
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
             if should_close:
                 conn.close()

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
from threading import Lock

from tqdm import tqdm
from dotenv import load_dotenv

from data.cleaning import clean_data
import data.utils as data_utils
from data.rate_limiter import RateLimiter
from data.settings import settings
import data.alpha_vantage_schema as avs

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """
    A comprehensive client for interacting with the Alpha Vantage API.

    Handles data fetching, parsing, caching, and rate limiting in a unified interface.
    """
    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[str] = None, db_conn: Optional[Union[DuckDBPyConnection, str]] = None, requests_per_minute: Optional[int] = None, requests_per_day: Optional[int] = None):
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
            self.conn = duckdb.connect(settings.get("db_name", "data/alpha_vantage.db"))
        
        self.db_lock = Lock()

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
                        print(data)
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
                     # If we executed the error check above and passed, it means we got a JSON that isn't an error.
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
                # Detect "No data" message specifically for HISTORICAL_OPTIONS
                if "message" in data and "No data for symbol" in str(data["message"]):
                    self.logger.info(f"Received 'No data' message: {data['message']}")
                    df = pd.DataFrame()
                    df.attrs["no_data_found"] = True
                    return df

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
                            df = df.assign(dt=pd.to_datetime(df["dt"], format="%Y-%m-%d"))
                            df.set_index("dt", inplace=True)
                            df.index.name = "dt"
                        
                        df = df.apply(pd.to_numeric, errors="coerce")
                        
                        # Rename columns to remove "1. ", "2. ", etc.
                        df.rename(
                            columns=lambda c: c.split(". ")[1] if ". " in c else c,
                            inplace=True,
                        )

                elif endpoint_name == "INSIDER_TRANSACTIONS":
                    # Insider Transactions
                    transactions = []
                    for k, v in data.items():
                         if isinstance(v, list):
                             transactions = v
                             break
                    
                    if transactions:
                        df = pd.DataFrame(transactions)
                    else:
                        self.logger.warning(f"No transaction list found in {endpoint_name} response")

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
                    # Clean common empty/null string representations
                    df.replace(["None", "none", "-", "0000-00-00", "null", "NULL"], [None, None, None, None, None, None], inplace=True)
                    
                    # Add timestamp for point-in-time reference if missing
                    if "dt" not in df.columns:
                        df["dt"] = pd.Timestamp.now("UTC")
                        
                    pass

            # Standardize datetime column
            date_candidates = [
                "timestamp", "fiscaldateending", "date", 
                "transactiondate", "transaction_date", "datadate",
                "reporteddate"
            ]
            
            for col in df.columns:
                if col.lower() in date_candidates:
                    df.rename(columns={col: "dt"}, inplace=True)
                    break 
            
            if "dt" not in df.columns and endpoint_name == "INSIDER_TRANSACTIONS":
                 # Fallback for insider transactions if date has weird name
                 for col in df.columns:
                     if "date" in col.lower():
                         df.rename(columns={col: "dt"}, inplace=True)
                         break 
            
            # Standardize symbol column
            if "ticker" in df.columns and "symbol" not in df.columns:
                df.rename(columns={"ticker": "symbol"}, inplace=True)

            if "dt" in df.columns:
                df = df.assign(dt=pd.to_datetime(df["dt"], format="%Y-%m-%d"))
                df.set_index("dt", inplace=True)
            elif df.index.name in ["timestamp", "fiscalDateEnding", "date"]:
                df.index.name = "dt"

            if endpoint_name in avs.SYMBOL_ENDPOINTS and "symbol" in params.keys():
                if not df.empty:
                    # Remove any existing columns that match "symbol" case-insensitively
                    # to avoid duplicates when we inject the param.
                    cols_to_drop = [c for c in df.columns if c.lower() == "symbol"]
                    if cols_to_drop:
                        df.drop(columns=cols_to_drop, inplace=True)
                    
                    df.loc[:, "symbol"] = params["symbol"]

            df.replace(["None", "none", "-", "0000-00-00"], [None, None, None, None], inplace=True)

        except Exception as e:
            self.logger.error(f"Error parsing data for {endpoint_name}: {e}")
            return pd.DataFrame()

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
                    df = df.assign(dt=pd.to_datetime(df["dt"], format="%Y-%m-%d"))
                    df.set_index("dt", inplace=True)
                    
            except Exception as utc_err:
                 self.logger.warning(f"Could not convert dt to UTC: {utc_err}")

        if not df.empty:
            # Normalize column names to camelCase
            df.rename(columns=lambda x: data_utils.normalize_to_camel_case(x), inplace=True)
            if df.index.name:
                df.index.name = data_utils.normalize_to_camel_case(df.index.name)
            # Apply cleaning overrides
            df = clean_data(endpoint_name, df)
            
        return df

    def _save_to_database(self, endpoint_name: str, df: pd.DataFrame, conn: Optional[DuckDBPyConnection] = None):
        """Save DataFrame to database using UPSERT logic."""
        if df.empty:
            return

        conn = conn or self.conn
        table_name = avs.ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper()
        
        with self.db_lock:
            try:
                schema_sql = avs.TABLE_SCHEMAS.get(table_name)
                if schema_sql:
                    create_sql = schema_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                    conn.execute(create_sql)
                else:
                    self.logger.error(f"No schema found for {table_name}")
                    return

                df_to_save = df.reset_index() if df.index.name == "dt" else df.copy()
                
                try:
                    db_cols = [c[0] for c in conn.execute(f"DESCRIBE {table_name}").fetchall()]
                    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]

                    valid_cols = [c for c in df_to_save.columns if c in db_cols]
                    if not valid_cols:
                        self.logger.warning(f"No matching columns for {table_name}")
                        return
                    df_to_save = df_to_save[valid_cols]
                    df_to_save = data_utils.enforce_schema(df_to_save, table_name)
                except Exception as e:
                    self.logger.error(f"Could not verify columns for {table_name}: {e}")
                    return

                pk_cols = avs.TABLE_PKS.get(table_name, [])

                if pk_cols:
                    initial_len = len(df_to_save)
                    valid_pk_cols = [c for c in pk_cols if c in df_to_save.columns]
                    
                    if valid_pk_cols:
                         df_to_save.dropna(subset=valid_pk_cols, inplace=True)
                         df_to_save.drop_duplicates(subset=valid_pk_cols, inplace=True)
                         
                    dropped = initial_len - len(df_to_save)
                    if dropped > 0:
                        self.logger.warning(f"Dropped {dropped} rows with invalid Primary Key/Date for {table_name}")

                    if df_to_save.empty:
                        return

                if not pk_cols:
                    conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save")
                    self.logger.info(f"Inserted {len(df_to_save)} rows into {table_name}")
                    return

                update_cols = [c for c in df_to_save.columns if c not in pk_cols]
                pk_str = ", ".join([f'"{c}"' for c in pk_cols])
                
                if update_cols:
                    update_str = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])
                    conflict_action = f"DO UPDATE SET {update_str}"
                else:
                    conflict_action = "DO NOTHING"

                try:
                    if conflict_action == "DO NOTHING":
                         conn.execute(f"INSERT OR IGNORE INTO {table_name} BY NAME SELECT * FROM df_to_save")
                    else:
                         conn.execute(f"""
                            INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save
                            ON CONFLICT ({pk_str}) {conflict_action}
                         """)
                    self.logger.info(f"Upserted {len(df_to_save)} rows into {table_name}")
                except duckdb.BinderException as e:
                    if "conflict target" in str(e).lower() and "constraint" in str(e).lower():
                        self.logger.warning(f"Constraint missing for {table_name}. Attempting to add UNIQUE INDEX or Manual Upsert.")
                        try:
                            index_name = f"idx_{table_name}_pk"
                            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name} ({pk_str})")
                            conn.execute(f"""
                                INSERT INTO {table_name} BY NAME SELECT * FROM df_to_save
                                ON CONFLICT ({pk_str}) {conflict_action}
                            """)
                            self.logger.info(f"Added index and upserted {len(df_to_save)} rows into {table_name}")
                        except Exception as index_err:
                            self.logger.info(f"Index creation failed ({index_err}). Using Delete-Insert fallback.")
                            
                            temp_table = f"temp_{table_name}_{int(time.time()*1000)}"
                            conn.register(temp_table, df_to_save)
                            
                            pk_tuple = f"({pk_str})"
                            conn.execute(f"DELETE FROM {table_name} WHERE {pk_tuple} IN (SELECT {pk_tuple} FROM {temp_table})")
                            conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM {temp_table}")
                            conn.unregister(temp_table)
                            self.logger.info(f"Manual upsert (delete-insert) completed for {table_name}")
                    else:
                        raise e

                except duckdb.ConstraintException as e:
                     self.logger.info(f"Ignored Constraint Error (Duplicate Key) for {table_name}: {e}")
                except Exception as e:
                     raise e

            except Exception as e:
                self.logger.error(f"Error in save flow for {endpoint_name} (Table: {table_name}): {e}", exc_info=True)

    def fetch_and_cache_data(self, endpoint_name: str, params: Dict, force_refresh: bool = False, min_required_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a single endpoint call.
        Checks DB first, then falls back to Parquet cache, then API.
        """
        if not force_refresh:
            # 1. Try DB first
            df = data_utils.get_from_db(self.conn, endpoint_name, params)
            if not df.empty:
                if min_required_date and "date" not in params:
                    latest_dt = df.index.max()
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
            filepath = data_utils.generate_filepath(self.data_dir, endpoint_name, params)
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath, engine='pyarrow')
                    self.logger.info(f"Loading from cache: {filepath}")
                    
                    if "dt" in df.columns:
                        df = df.assign(dt=pd.to_datetime(df["dt"], format="%Y-%m-%d"))
                        df.set_index("dt", inplace=True)
                        df.index.name = "dt"

                    if "date" in params:
                        self._save_to_database(endpoint_name, params, df)
                        return df

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
                            self._save_to_database(endpoint_name, df)
                            return df
                    else:
                        return df
                except Exception as e:
                    self.logger.error(f"Error reading parquet cache: {e}")

        # 3. Fetch fresh data (API)
        raw_data = self._fetch_data(endpoint_name, params)
        df = self._parse_response(endpoint_name, raw_data, params)

        if df.empty and not df.attrs.get("no_data_found"):
            return df

        # 4. Save to Parquet
        filepath = data_utils.generate_filepath(self.data_dir, endpoint_name, params)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, engine='pyarrow')
            self.logger.info(f"Saved to cache: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving parquet cache: {e}")

        # 5. Save to Database
        self._save_to_database(endpoint_name, df)

        return df

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

        all_dfs = []
        self.logger.info(f"Starting execution of {len(tasks)} tasks sequentially.")

        min_required_datetime = pd.to_datetime(end_date).tz_localize("UTC") if end_date else None

        # Progress bar
        with tqdm(total=len(tasks), unit="req", desc="Fetching Data") as pbar:
            for endpoint_name, params in tasks:
                try:
                    df = self.fetch_and_cache_data(
                        endpoint_name, 
                        params, 
                        force_refresh,
                        min_required_datetime
                    )
                    
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
            symbols = symbols or data_utils.read_stock_symbols()
            endpoints = endpoints or data_utils.get_endpoints()
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            start_date = start_date or (pd.to_datetime(end_date) - pd.DateOffset(years=15)).strftime("%Y-%m-%d")

            # --- Build Task List ---
            tasks = []
            
            # 1. Symbol-specific tasks
            avs.SYMBOL_ENDPOINTS = {k: v for k, v in endpoints.items() if k in avs.SYMBOL_ENDPOINTS}
            
            for symbol in symbols:
                for endpoint_name, params in avs.SYMBOL_ENDPOINTS.items():
                    # Create a copy of params
                    task_params = params.copy()
                    task_params["symbol"] = symbol
                    
                    # Only endpoint with HISTORICAL_OPTIONS
                    if "date" in params:
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
            avs.MACRO_ENDPOINTS_map = {k: v for k, v in endpoints.items() if k in avs.MACRO_ENDPOINTS}
            for endpoint_name, params in avs.MACRO_ENDPOINTS_map.items():
                tasks.append((endpoint_name, params))

            return self._execute_tasks(tasks, force_refresh, start_date, end_date)

        except Exception as e:
            self.logger.error(f"Error in get_data: {e}", exc_info=True)
            return pd.DataFrame()

    def _scan_local_cache(self) -> dict:
        """
        Returns a dictionary of all valid parquet file paths in the data directory.
        Keys are unresolved filenames, Values are full resolved paths.
        """
        files = {}
        base_dir = Path(self.data_dir, "files")
        if not base_dir.exists():
            return files
            
        for p in base_dir.rglob("*.parquet"):
            files[str(p)] = True
            
        return files

    def _get_detailed_db_state(self, endpoints: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Efficiently retrieves the state of the database to minimize queries.
        Returns:
            Dict[Table, Dict[Symbol/Key, State]]
        """
        state = {}
        # Identify relevant tables
        tables = set()
        for ep in endpoints:
            tables.add(avs.ENDPOINT_TO_TABLE_MAP.get(ep, ep).upper())
        
        # Add MACRO if needed
        if any(ep in avs.MACRO_ENDPOINTS for ep in endpoints):
            tables.add("MACRO")

        try:
            existing_tables_df = self.conn.execute("SHOW TABLES").df()
            if existing_tables_df.empty:
                return {}
            existing_tables = set(existing_tables_df['name'].str.upper())
        except Exception:
            return {}

        for table in tables:
            if table not in existing_tables:
                continue
            
            try:
                # Check columns to decide query strategy
                cols = [c[0] for c in self.conn.execute(f"DESCRIBE {table}").fetchall()]
                
                # Case 1: HISTORICAL_OPTIONS (Dense Date Data)
                if table == "HISTORICAL_OPTIONS":
                     # For options, we need the exact set of dates per symbol
                     # Fetching all might be heavy, but it's the only way to accurately diff.
                     # limit columns to reduce I/O
                     query = f"SELECT symbol, dt FROM {table}"
                     df = self.conn.execute(query).df()
                     if not df.empty:
                         df['dt'] = df['dt'].astype(str)
                         # dict: symbol -> set of date strings
                         state[table] = df.groupby('symbol')['dt'].apply(set).to_dict()
                     continue

                # Case 2: MACRO
                if table == "MACRO":
                    state[table] = {}
                    macro_eps = [ep for ep in endpoints if avs.ENDPOINT_TO_TABLE_MAP.get(ep, ep).upper() == "MACRO"]
                    for ep in macro_eps:
                         col = ep.lower()
                         if col in cols:
                             res = self.conn.execute(f"SELECT MAX(dt) FROM {table} WHERE {col} IS NOT NULL").fetchone()
                             if res and res[0]:
                                 dt = pd.to_datetime(res[0])
                                 if dt.tz is None: dt = dt.tz_localize("UTC")
                                 else: dt = dt.tz_convert("UTC")
                                 state[table][ep] = dt
                    continue

                # Case 3: Standard Time Series / Overview (Max Date per Symbol)
                if ("dt" in cols or "date" in cols) and "symbol" in cols:
                    dt_col = "dt" if "dt" in cols else "date"
                    query = f"SELECT symbol, MAX({dt_col}) as max_dt FROM {table} GROUP BY symbol"
                    df = self.conn.execute(query).df()
                    if not df.empty:
                        df["max_dt"] = pd.to_datetime(df["max_dt"])
                        # Handle TZ
                        if df["max_dt"].dt.tz is None:
                             df["max_dt"] = df["max_dt"].dt.tz_localize("UTC")
                        else:
                             df["max_dt"] = df["max_dt"].dt.tz_convert("UTC")
                        state[table] = dict(zip(df["symbol"], df["max_dt"]))
                    continue

            except Exception as e:
                self.logger.warning(f"Error getting state for {table}: {e}")

        return state

    def smart_update(
        self,
        symbols: Optional[List[str]] = None,
        endpoints: Optional[Dict[str, Dict]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        workers: Optional[int] = None
    ) -> None:
        """
        Efficiently updates data using parallel fetching and bulk DB checking.
        """
        symbols = symbols or data_utils.read_stock_symbols()
        endpoints = endpoints or data_utils.get_endpoints()
        max_workers = workers or settings.get("workers", 8)
        
        # Defaults
        now_utc = pd.Timestamp.now("UTC")
        end_date_dt = pd.to_datetime(end_date).tz_localize("UTC") if end_date else now_utc
        s_date_default = pd.to_datetime(start_date or (now_utc - pd.DateOffset(years=2)))
        if s_date_default.tz is None:
            s_date_default = s_date_default.tz_localize("UTC")
        else:
            s_date_default = s_date_default.tz_convert("UTC")

        self.logger.info("Phase 1: Inventorying Local Cache & Database State...")
        
        # Parallelize inventory steps
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_local = executor.submit(self._scan_local_cache)
            future_db = executor.submit(self._get_detailed_db_state, endpoints)
            
            local_cache_files = future_local.result()
            db_state = future_db.result()

        self.logger.info(f"Inventory complete. Found {len(local_cache_files)} local files.")

        api_tasks = []
        db_tasks = []

        # Identify Tasks
        self.logger.info("Phase 2: Calculating Tasks...")
        
        sym_endpoints = {k: v for k, v in endpoints.items() if k in avs.SYMBOL_ENDPOINTS}
        
        # Prepare date range for options once
        full_range = pd.date_range(s_date_default, end_date_dt).normalize()
        # Filter out weekends (Saturday=5, Sunday=6)
        full_range = full_range[full_range.dayofweek < 5]
        
        # Filter out market holidays
        holidays = data_utils.get_market_holidays(s_date_default, end_date_dt)
        # Convert holidays to tz-aware to match full_range if needed, or normalize both
        # full_range is normalized (midnight). Holidays are usually midnight but naive.
        # Let's ensure comparison works.
        full_range = full_range[~full_range.isin(holidays)]
        
        full_range_strs = set(d.strftime("%Y-%m-%d") for d in full_range)

        # Helper to check if update needed based on dates
        def is_stale(ep_name, last_date):
            if not last_date: return True
            
            if "INTRADAY" in ep_name:
                 return (end_date_dt - last_date).total_seconds() > 3600 * 4
            
            if "DAILY" in ep_name:
                # If last date is today or later, we are good
                return last_date.date() < end_date_dt.date()
            
            if ep_name in avs.FUNDAMENTAL_ENDPOINTS:
                # 92-day buffer
                return (end_date_dt - last_date).days >= 92
            
            return True

        with tqdm(total=len(symbols), unit="req", desc="Analyzing Symbols") as pbar:
            for sym in symbols:
                for ep_name, base_params in sym_endpoints.items():
                    table = avs.ENDPOINT_TO_TABLE_MAP.get(ep_name, ep_name).upper()
                    
                    # Special Case: HISTORICAL_OPTIONS (Date Range)
                    if "date" in base_params:
                        # We need to cover full_range_strs existing in DB
                        existing_db_dates = db_state.get(table, {}).get(sym, set())
                        needed_dates = full_range_strs - existing_db_dates
                        
                        for d_str in needed_dates:
                            p = base_params.copy()
                            p["symbol"] = sym
                            p["date"] = d_str
                            
                            # Check if we have it locally
                            fp = data_utils.generate_filepath(self.data_dir, ep_name, p)
                            fp_str = str(fp)
                            
                            if fp_str in local_cache_files:
                                # Have file, missing in DB -> DB Task
                                db_tasks.append((ep_name, p, fp_str))
                            else:
                                # Missing file -> API Task
                                api_tasks.append((ep_name, p, fp_str))
                        continue
                    
                    # Check DB State first
                    last_dt = db_state.get(table, {}).get(sym)
                    
                    if not is_stale(ep_name, last_dt):
                        continue

                    # Standard Case
                    p = base_params.copy()
                    p["symbol"] = sym
                    
                    # Check if we have it locally
                    fp = data_utils.generate_filepath(self.data_dir, ep_name, p)
                    fp_str = str(fp)
                    
                    if fp_str in local_cache_files:
                        # Have file, missing in DB -> DB Task
                        db_tasks.append((ep_name, p, fp_str))
                    else:
                        # Missing file -> API Task
                        api_tasks.append((ep_name, p, fp_str))
                    
                pbar.update(1)
                pbar.set_postfix_str(f"Last: {sym}", refresh=False)

        # Macro Tasks
        macro_endpoints = {k: v for k, v in endpoints.items() if k in avs.MACRO_ENDPOINTS}
        for ep_name, params in macro_endpoints.items():
             # Basic check
             table = "MACRO"
             last_dt = db_state.get(table, {}).get(ep_name)
             if is_stale(ep_name, last_dt):
                 fp = data_utils.generate_filepath(self.data_dir, ep_name, params)
                 api_tasks.append((ep_name, params, str(fp)))
        
        self.logger.info(f"Tasks Identified: {len(db_tasks)} DB Ingestions, {len(api_tasks)} API Fetches.")

        # Phase 3: Execution
        
        # Part A: DB Ingestion (CPU/Disk Bound - mostly massive inserts)
        if db_tasks:
            self.logger.info(f"Processing {len(db_tasks)} missing DB records from local cache...")
            
            # Group tasks by endpoint
            tasks_by_endpoint = {}
            for ep, p, fp in db_tasks:
                if ep not in tasks_by_endpoint:
                    tasks_by_endpoint[ep] = []
                tasks_by_endpoint[ep].append(fp)

            with tqdm(total=len(db_tasks), unit="rows", desc="DB Ingestion") as pbar:
                for ep, file_paths in tasks_by_endpoint.items():
                    dfs_to_concat = []
                    for fp in file_paths:
                        try:
                            df = pd.read_parquet(fp)
                            if not df.empty and len(df.dropna()) > 0:
                                dfs_to_concat.append(df)
                        except Exception as e:
                            self.logger.error(f"Failed to load cached {fp}: {e}")
                        pbar.update(1)
                    
                    if dfs_to_concat:
                        try:
                            combined_df = pd.concat(dfs_to_concat)
                            self._save_to_database(ep, combined_df)
                        except Exception as e:
                            self.logger.error(f"Failed to save batch for {ep}: {e}")

        # Part B: API Fetching
        if not api_tasks:
            self.logger.info("All data up to date.")
            return

        self.logger.info(f"Starting {len(api_tasks)} API fetches (workers={max_workers})...")
        
        # Execute tasks sequentially using _execute_tasks
        # Filter api_tasks to (endpoint, params) tuples expected by _execute_tasks
        seq_tasks = [(t[0], t[1]) for t in api_tasks]
        self._execute_tasks(seq_tasks, force_refresh=True)

        self.logger.info("Smart update completed successfully.")

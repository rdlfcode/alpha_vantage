"""
Alpha Vantage Client

A comprehensive client for interacting with the Alpha Vantage API.
Handles data fetching, parsing, caching, and rate limiting in a unified interface.
"""

import os
import requests
import pandas as pd
import duckdb
from duckdb import DuckDBPyConnection

import logging
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, List, Union
from dotenv import load_dotenv

from alpha_vantage_schema import BASE_URL, SYMBOL_ENDPOINTS, MACRO_ENDPOINTS, TABLE_SCHEMAS
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
    
    def __init__(self, api_key: Optional[str] = None,
                 data_dir: Optional[str] = None, 
                 db_path: Optional[str] = None,
                 requests_per_minute: Optional[int] = None, 
                 requests_per_day: Optional[int] = None):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key. If None, will try to get from environment.
            data_dir: Directory for caching data files
            db_path: Path to SQLite database for data storage
            requests_per_minute: Rate limit for requests per minute
            requests_per_day: Rate limit for requests per day
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter.")
        
        # Use settings with fallbacks
        self.data_dir = Path(data_dir or settings.get("data_dir", "data"))
        self.db_path = db_path or settings.get("db_path", "data/alpha_vantage.db")
        
        rpm = requests_per_minute or settings.get("AlphaVantageRPM", 75)
        rpd = requests_per_day or settings.get("AlphaVantageRPD", 25)
        
        self.rate_limiter = RateLimiter(requests_per_minute=rpm, requests_per_day=rpd)
        self.logger = logger
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
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
            response = requests.get(BASE_URL, params=full_params, timeout=30)
            response.raise_for_status()
            
            # Check for API errors
            if "Information" in response.text or "Note" in response.text:
                self.logger.warning(f"API returned an error or note for {endpoint_name}: {response.text[:200]}...")
                return None
                
            # Return appropriate format
            if params.get("datatype", "json") == "json":
                return response.json()
            else:
                return response.text
                
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
                df = pd.read_csv(StringIO(data))

            elif isinstance(data, dict):
                # Parse JSON response
                if any("Time Series" in k for k in data.keys()):
                    # Time series data
                    time_series_key = next(k for k in data.keys() if "Time Series" in k)
                    time_series_data = data[time_series_key]
                    if isinstance(time_series_data, dict):
                        df = pd.DataFrame.from_dict(time_series_data, orient="index")
                        df.index = pd.to_datetime(df.index)
                        df = df.apply(pd.to_numeric, errors="coerce")

                elif any("data" in k.lower() for k in data.keys()):
                    # Economic indicators and commodities
                    data_key = next(k for k in data.keys() if "data" in k.lower())
                    time_series_data = data[data_key]
                    if isinstance(time_series_data, list):
                        df = pd.DataFrame(time_series_data)
                        # Convert numeric columns
                        numeric_cols = df.select_dtypes(include=['object']).columns
                        for col in numeric_cols:
                            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

                elif any(k in data for k in ["quarterlyReports", "annualReports"]):
                    # Financial statements
                    reports_key = "quarterlyReports" if "quarterlyReports" in data else "annualReports"
                    df = pd.DataFrame(data[reports_key])

                else:
                    # Fallback for other structures
                    df = pd.DataFrame.from_dict(data, orient="index").transpose()

            # Standardize datetime column
            for col_name in ["timestamp", "fiscalDateEnding", "date", "transactionDate"]:
                if col_name in df.columns:
                    df.rename(columns={col_name: "dt"}, inplace=True)
                    break
            
            if "dt" in df.columns:
                df.loc[:, "dt"] = pd.to_datetime(df["dt"])
                df.set_index("dt", inplace=True)
            elif df.index.name in ["timestamp", "fiscalDateEnding", "date"]:
                df.index.name = "dt"

            if endpoint_name in SYMBOL_ENDPOINTS and "symbol" in params.keys():
                df["symbol"] = params["symbol"]

        except Exception as e:
            self.logger.error(f"Error parsing data for {endpoint_name}: {e}")
            return pd.DataFrame()
            
        return df

    def _fetch_and_cache_data(self, endpoint_name: str, params: Dict, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a single endpoint call.
        """
        # Generate cache filename
        filepath = self._generate_filepath(endpoint_name, params)
        
        # Return cached data if available and not forcing refresh
        if filepath.exists() and not force_refresh:
            self.logger.info(f"Loading from cache: {filepath}")
            try:
                return pd.read_parquet(filepath)
            except Exception as e:
                self.logger.warning(f"Error loading cached data: {e}. Fetching fresh data.")
        
        # Fetch fresh data
        raw_data = self._fetch_data(endpoint_name, params)
        df = self._parse_response(endpoint_name, raw_data, params)

        # Cache the data if not empty
        if not df.empty:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath)
            self.logger.info(f"Saved to cache: {filepath}")
            
            # Also save to database
            self._save_to_database(endpoint_name, params, df)

        return df

    def _save_to_database(self, endpoint_name: str, params: Dict, df: pd.DataFrame, conn: Optional[DuckDBPyConnection] = None):
        """
        Save DataFrame to SQLite database using the correct schema for each endpoint.
        """
        if df.empty:
            self.logger.warning(f"No data to save for {endpoint_name} with params {params}")
            return

        table_name = endpoint_name.upper()
        conn = conn or duckdb.connect(self.db_path)
        
        try:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            conn.close()
            self.logger.info(f"Saved {len(df)} rows to {table_name}")

        except Exception as e:
            match e:
                # Table doesn't exist
                case duckdb.CatalogException:
                    conn = duckdb.connect(self.db_path)
                    conn.execute(TABLE_SCHEMAS.get(endpoint_name))
                case _:
                    self.logger.error(f"Error saving to database for {endpoint_name}: {e}")

    def get_data(self, 
                 symbols: Optional[List[str]] = None,
                 endpoints: Optional[Dict[str, Dict]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
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
            symbol_endpoints = {k: v for k, v in endpoints.items() if k in SYMBOL_ENDPOINTS}
            for symbol in symbols:
                for endpoint_name, params in symbol_endpoints.items():
                    params.update({"symbol": symbol})
                    df = self._fetch_and_cache_data(endpoint_name, params, force_refresh)
                    if not df.empty:
                        all_dfs.append(df)

            # Process macro-economic endpoints
            macro_endpoints_map = {k: v for k, v in endpoints.items() if k in MACRO_ENDPOINTS}
            for endpoint_name, params in macro_endpoints_map.items():
                df = self._fetch_and_cache_data(endpoint_name, params, force_refresh)
                if not df.empty:
                    all_dfs.append(df)

            if not all_dfs:
                self.logger.warning("No data fetched for the given symbols and endpoints.")
                return pd.DataFrame()

            # Combine all DataFrames
            combined_df = pd.concat(all_dfs, sort=False)

            # Filter by date range if specified
            if 'date' in combined_df.index.names:
                if start_date:
                    combined_df = combined_df[combined_df.index >= start_date]
                if end_date:
                    combined_df = combined_df[combined_df.index <= end_date]
            
            self.logger.info(f"Successfully built dataset with {len(combined_df)} rows.")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error in get_data: {e}", exc_info=True)
            return pd.DataFrame()

    def get_dataset_from_db(self,
                            sql_query: str,
                            conn: Optional[DuckDBPyConnection] = None) -> pd.DataFrame:
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
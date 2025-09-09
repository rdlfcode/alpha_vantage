import os
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, List
from datetime import datetime
from settings import settings
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import alpha_vantage_api as ava
import logging
from rate_limiter import RateLimiter
import duckdb

load_dotenv()

# --- Logging Setup ---
log_settings = settings.get("logging", {})
logging.basicConfig(
    level=log_settings.get("level", "INFO"),
    format=log_settings.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)

# --- Rate Limiter ---
rate_limiter = RateLimiter(
    requests_per_minute=settings["AlphaVantageRPM"],
    requests_per_day=settings["AlphaVantageRPD"]
)


def read_stock_symbols(file_path: str = "stocks.txt") -> List[str]:
    """Read stock symbols from a file, one per line."""
    try:
        with open(file_path, "r") as f:
            return sorted(list(set([line.strip() for line in f if line.strip()])))
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Using default symbols.")
        return ["IBM", "AAPL"]


def generate_filename(endpoint_name, params) -> str:
    """Generates a filename from the endpoint and its parameters."""
    filename_parts = [endpoint_name]

    # Create a copy of params to modify
    params_copy = params.copy()

    # Extract and remove list-based parameters to handle them separately if needed
    list_params = {k: v for k, v in params_copy.items() if isinstance(v, list)}
    for k in list_params:
        del params_copy[k]

    # Handle symbol-like parameters for clarity in the filename
    symbol_like_keys = ["symbol", "keywords", "tickers", "from_symbol", "to_symbol"]
    symbol_value = None
    for key in symbol_like_keys:
        if key in params_copy:
            symbol_value = params_copy.pop(key)
            filename_parts.insert(1, f"symbol_{symbol_value}")
            break

    # Add remaining static parameters
    for key, value in sorted(params_copy.items()):
        if key != "apikey" and isinstance(value, str):
            filename_parts.append(f"{key}_{value}")

    return "_".join(filename_parts)


def cache_to_parquet(data_dir):
    """
    Decorator to cache the output of a function to a Parquet file.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(endpoint_name, params, *args, **kwargs):
            filepath = Path(data_dir) / (
                generate_filename(endpoint_name, params) + ".parquet"
            )

            if filepath.exists():
                logger.info(f"Loading from cache: {filepath}")
                return pd.read_parquet(filepath)

            df: pd.DataFrame = func(endpoint_name, params, *args, **kwargs)

            if not df.empty:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(filepath)
                logger.info(f"Saved to cache: {filepath}")

            return df
        return wrapper
    return decorator


def fetch_data(endpoint_name, params) -> dict | str | None:
    """Fetches data from a single Alpha Vantage endpoint."""
    rate_limiter.wait_if_needed()
    full_params = {
        "function": endpoint_name,
        "apikey": os.environ["ALPHA_VANTAGE_API_KEY"],
        **params,
    }
    logger.info(f"Fetching data for {endpoint_name} with params: {params}")
    try:
        response = requests.get(ava.BASE_URL, params=full_params)
        response.raise_for_status()  # Raise an exception for bad status codes
        # Alpha Vantage sometimes returns an error message in JSON even for 200 status
        if "Information" in response.text or "Note" in response.text:
            logger.warning(f"API returned an error or note for {endpoint_name}: {response.text}")
            return None

        if params.get("datatype", "json") == "json":
            return response.json()
        else:
            return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for {endpoint_name}: {e}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"No data returned for {endpoint_name}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred for {endpoint_name}: {e}")
        return None


@cache_to_parquet(settings["data_dir"])
def fetch_data_df(endpoint_name, params) -> pd.DataFrame:
    """Fetches DataFrame from a single Alpha Vantage endpoint."""
    data = fetch_data(endpoint_name, params)

    if data is None:
        return pd.DataFrame()

    df = pd.DataFrame()
    # Handle CSV data
    if params.get("datatype", "json") == "csv" and isinstance(data, str):
        try:
            df = pd.read_csv(StringIO(data))
            # Set timestamp/date as index if it exists
            if "timestamp" in df.columns:
                df.rename(columns={"timestamp": "date"}, inplace=True)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            logger.debug(f"CSV data for {endpoint_name} converted to DataFrame with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error parsing CSV data for {endpoint_name}: {e}")
            df = pd.DataFrame()

    # Handle JSON data
    elif isinstance(data, dict):
        try:
            # Handle time series data
            if any("data" in k for k in data.keys()):
                time_series_key = next(k for k in data.keys() if "data" in k)
                time_series_data = data[time_series_key]
                if isinstance(time_series_data, dict):
                    df = pd.DataFrame.from_dict(time_series_data, orient="index")
                    df = df.apply(pd.to_numeric, errors="coerce")
                    if "transaction_date" in df.columns:
                        df.rename(columns={"transaction_date": "date"}, inplace=True)

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)

            # Handle fundamental data (quarterly/annual reports)
            elif any(k in data for k in ["quarterlyReports", "annualReports"]):
                reports_key = (
                    "quarterlyReports"
                    if "quarterlyReports" in data
                    else "annualReports"
                )
                if isinstance(data.get(reports_key), list):
                    df = pd.DataFrame(data[reports_key])
                    if "fiscalDateEnding" in df.columns:
                        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
                        df.set_index("fiscalDateEnding", inplace=True)
                    numeric_cols = df.select_dtypes(include=["object"]).columns
                    df[numeric_cols] = (
                        df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                    )

            # Handle other JSON structures
            else:
                # Attempt to build a DataFrame, assuming a simple structure
                df = pd.DataFrame.from_dict(data, orient="index").transpose()
                if "timestamp" in df.columns:
                    df.rename(columns={"timestamp": "date"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
            logger.debug(f"JSON data for {endpoint_name} converted to DataFrame with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error parsing JSON data for {endpoint_name}: {e}")
            df = pd.DataFrame()
    return df


def download_all_data(endpoints: dict, stock_symbols: list, force_refresh: bool = False):
    """
    Iterates through all endpoints and symbols, downloading data and saving it to Parquet files.
    """
    data_dir = Path(settings["data_dir"])
    data_dir.mkdir(exist_ok=True)

    macro_endpoints = {n: p for n, p in endpoints.items() if "symbol" not in p}
    symbol_endpoints = {n: p for n, p in endpoints.items() if "symbol" in p}

    # Download macro data
    for name, params in macro_endpoints.items():
        filepath = data_dir / (generate_filename(name, params) + ".parquet")
        if not filepath.exists() or force_refresh:
            logger.info(f"Fetching '{name}'...")
            df = fetch_data_df(name, params)
            if not df.empty:
                # No need to save here as the decorator does it
                pass
        else:
            logger.info(f"'{name}' data already cached at {filepath}. Skipping download.")

    # Download data for each symbol
    for symbol in stock_symbols:
        for name, params in symbol_endpoints.items():
            p = params.copy()
            p["symbol"] = symbol
            filepath = data_dir / (generate_filename(name, p) + ".parquet")
            if not filepath.exists() or force_refresh:
                logger.info(f"Fetching '{name}' for symbol '{symbol}'...")
                df = fetch_data_df(name, p)
                if not df.empty:
                    # No need to save here as the decorator does it
                    pass
            else:
                logger.info(
                    f"'{name}' for '{symbol}' already cached at {filepath}. Skipping download."
                )


def get_dataset(sql_query: str) -> pd.DataFrame:
    """
    Executes a SQL query using DuckDB on the cached Parquet files and returns a DataFrame.
    """
    try:
        con = duckdb.connect(database=settings["db_path"], read_only=False)
        logger.info(f"Executing query:\n{sql_query}")
        result_df = con.execute(sql_query).fetchdf()
        con.close()
        return result_df
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}")
        return pd.DataFrame()


def fetch_alpha_vantage_data(symbols: Optional[List[str]] = None, 
                        endpoints: Optional[Dict[str, Dict]] = None, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        force_refresh: bool = False) -> pd.DataFrame:
    """
    Main function to fetch and process Alpha Vantage data.
    
    Args:
        symbols: List of stock symbols to fetch data for. If None, uses STOCK_SYMBOLS.
        endpoints: Dictionary of endpoints and their parameters. If None, uses ENDPOINTS.
        start_date: Start date for the data in YYYY-MM-DD format. If None, fetches all available data.
        end_date: End date for the data in YYYY-MM-DD format. If None, uses current date.
        force_refresh: Whether to force refresh all data regardless of what's cached.
    
    Returns:
        DataFrame containing the combined dataset with all requested data.
    """
    # Set defaults
    symbols = symbols or STOCK_SYMBOLS
    endpoints = endpoints or ENDPOINTS
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    
    # Create data directory if it doesn't exist
    data_dir = Path(settings["data_dir"])
    data_dir.mkdir(exist_ok=True)
    
    # Filter endpoints to only include those specified in our categories
    valid_endpoints = {k: v for k, v in endpoints.items() 
                      if k in ava.SYMBOL_ENDPOINTS + ava.MACRO_ENDPOINTS}
    
    # Download all required data
    download_all_data(valid_endpoints, symbols, force_refresh=force_refresh)
    
    # Build and return the combined dataset
    return build_combined_dataset(symbols, valid_endpoints, start_date, end_date)


def build_combined_dataset(symbols: List[str], endpoints: Dict[str, Dict], 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Build a combined dataset from downloaded data files with optional date filtering.
    
    Args:
        symbols: List of stock symbols to include.
        endpoints: Dictionary of endpoints and their parameters.
        start_date: Optional start date filter in YYYY-MM-DD format.
        end_date: Optional end date filter in YYYY-MM-DD format.
    
    Returns:
        DataFrame containing the combined dataset.
    """
    con = duckdb.connect(database=settings["db_path"], read_only=False)
    data_dir = Path(settings["data_dir"])
    
    # Build query components for each data type
    ctes = []
    base_tables = []
    
    # Process symbol-specific data
    for symbol in symbols:
        for name in [ep for ep in endpoints if ep in ava.SYMBOL_ENDPOINTS]:
            params = endpoints[name].copy()
            params["symbol"] = symbol
            file_path = data_dir / (generate_filename(name, params) + ".parquet")
            if file_path.exists():
                table_name = f"{name}_{symbol}".lower()
                cte = f"""
                    {table_name} AS (
                        SELECT *,
                        '{symbol}' as symbol
                        FROM read_parquet('{file_path}')
                        WHERE 1=1
                        {f"AND date >= '{start_date}'" if start_date else ""}
                        {f"AND date <= '{end_date}'" if end_date else ""}
                    )
                """
                ctes.append(cte.strip())
                if name == "TIME_SERIES_DAILY":
                    base_tables.append(table_name)
                    
    # Process macro endpoints
    for name in [ep for ep in endpoints if ep in ava.MACRO_ENDPOINTS]:
        params = endpoints[name]
        file_path = data_dir / (generate_filename(name, params) + ".parquet")
        if file_path.exists():
            table_name = name.lower()
            cte = f"""
                {table_name} AS (
                    SELECT *
                    FROM read_parquet('{file_path}')
                    WHERE 1=1
                    {f"AND date >= '{start_date}'" if start_date else ""}
                    {f"AND date <= '{end_date}'" if end_date else ""}
                )
            """
            ctes.append(cte.strip())
            
    # Build the main query
    if not base_tables:
        logger.warning("No time series data found for any symbol")
        return pd.DataFrame()
    
    # Start with time series data as base
    query = "WITH " + ",\n".join(ctes)
    
    # Join all symbol data first
    query += f"""
        , combined_base AS (
            SELECT * FROM {base_tables[0]}
            {' UNION ALL '.join(f'SELECT * FROM {t}' for t in base_tables[1:])}
        )
        , enriched_symbols AS (
            SELECT cb.*
            FROM combined_base cb
    """
    
    # Join additional symbol data
    for symbol in symbols:
        for name in [ep for ep in endpoints if ep in ava.SYMBOL_ENDPOINTS and ep != "TIME_SERIES_DAILY"]:
            table_name = f"{name}_{symbol}".lower()
            if f"{table_name} AS" in query:  # Check if we have this data
                query += f"""
                    LEFT JOIN {table_name} USING (date, symbol)
                """
                
    query += ")"
    
    # Join macro data last
    query += """
        SELECT es.*
    """
    
    # Add macro columns
    macro_joins = []
    for name in [ep for ep in endpoints if ep in ava.MACRO_ENDPOINTS]:
        table_name = name.lower()
        if f"{table_name} AS" in query:  # Check if we have this data
            query += f", {table_name}.*"
            macro_joins.append(f"""
                LEFT JOIN {table_name}
                ON es.date >= {table_name}.date
                AND es.date < LEAD({table_name}.date, 1) OVER (ORDER BY {table_name}.date)
            """)
    
    query += " FROM enriched_symbols es"
    query += " ".join(macro_joins)
    query += " ORDER BY date DESC, symbol;"
    
    try:
        result_df = get_dataset(query)
        con.close()
        
        if not result_df.empty:
            output_path = data_dir / "final_dataset.parquet"
            result_df.to_parquet(output_path)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error building combined dataset: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # This section is just for testing/demonstration

    # Read stock symbols from the configuration file
    STOCK_SYMBOLS = read_stock_symbols()

    # Define endpoint parameters (None for symbol, will be filled in later)
    ENDPOINTS = {
        "TIME_SERIES_DAILY": {"symbol": None, "outputsize": "full", "datatype": "csv"},
        "INSIDER_TRANSACTIONS": {"symbol": None},
        "INCOME_STATEMENT": {"symbol": None},
        "BALANCE_SHEET": {"symbol": None},
        "CASH_FLOW": {"symbol": None},
        "EARNINGS": {"symbol": None},
        "WTI": {"interval": "daily", "datatype": "csv"},
        "BRENT": {"interval": "daily", "datatype": "csv"},
        "NATURAL_GAS": {"interval": "daily", "datatype": "csv"},
        "COPPER": {"interval": "monthly", "datatype": "csv"},
        "ALUMINUM": {"interval": "monthly", "datatype": "csv"},
        "WHEAT": {"interval": "monthly", "datatype": "csv"},
        "CORN": {"interval": "monthly", "datatype": "csv"},
        "COTTON": {"interval": "monthly", "datatype": "csv"},
        "SUGAR": {"interval": "monthly", "datatype": "csv"},
        "COFFEE": {"interval": "monthly", "datatype": "csv"},
        "ALL_COMMODITIES": {"interval": "monthly", "datatype": "csv"},
        "REAL_GDP": {"interval": "annual", "datatype": "csv"},
        "REAL_GDP_PER_CAPITA": {"datatype": "csv"},
        "TREASURY_YIELD": {"interval": "daily", "maturity": "10year"},
        "FEDERAL_FUNDS_RATE": {"interval": "daily", "datatype": "csv"},
        "CPI": {"interval": "monthly", "datatype": "csv"},
        "INFLATION": {"datatype": "csv"},
        "RETAIL_SALES": {"datatype": "csv"},
        "DURABLES": {"datatype": "csv"},
        "UNEMPLOYMENT": {"datatype": "csv"},
        "NONFARM_PAYROLL": {"datatype": "csv"},
    }

    df = fetch_alpha_vantage_data(symbols=STOCK_SYMBOLS, endpoints=ENDPOINTS, force_refresh=False)

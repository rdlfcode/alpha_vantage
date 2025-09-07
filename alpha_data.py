import os
from pathlib import Path
from functools import wraps
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
    requests_per_day=settings["AlphaVantageRPD"],
)


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


# Read stock symbols from a file
try:
    with open("stocks.txt", "r") as f:
        STOCK_SYMBOLS = sorted(list(set([line.strip() for line in f if line.strip()])))
except FileNotFoundError:
    logger.error("stocks.txt not found. Using default symbols.")
    STOCK_SYMBOLS = ["IBM", "AAPL"]

# Define non-premium endpoints and their parameters
ENDPOINTS = {
    "TIME_SERIES_DAILY": {
        "symbol": STOCK_SYMBOLS,
        "outputsize": "full",
        "datatype": "csv",
    },
    "INSIDER_TRANSACTIONS": {"symbol": STOCK_SYMBOLS},
    "INCOME_STATEMENT": {"symbol": STOCK_SYMBOLS},
    "BALANCE_SHEET": {"symbol": STOCK_SYMBOLS},
    "CASH_FLOW": {"symbol": STOCK_SYMBOLS},
    "EARNINGS": {"symbol": STOCK_SYMBOLS},
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

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path(settings["data_dir"])
    data_dir.mkdir(exist_ok=True)

    # Phase 1: Download all required data, with an option to force a refresh
    # Set force_refresh=True to re-download all data
    download_all_data(ENDPOINTS, STOCK_SYMBOLS, force_refresh=False)

    # Phase 2: Build dataset using DuckDB
    # This query replicates the logic of the original get_timeseries_data function.
    con = duckdb.connect(database=settings["db_path"], read_only=False)

    # Identify macro and symbol endpoints from the main ENDPOINTS dictionary
    macro_endpoints = {n: p for n, p in ENDPOINTS.items() if "symbol" not in p}
    symbol_endpoints = {n: p for n, p in ENDPOINTS.items() if "symbol" in p}

    # --- Symbol Data CTEs ---
    symbol_ctes = []
    for symbol in STOCK_SYMBOLS:
        for name, params in symbol_endpoints.items():
            p = params.copy()
            p["symbol"] = symbol
            file_path = data_dir / (generate_filename(name, p) + ".parquet")
            if file_path.exists():
                # Use symbol as the CTE name to avoid duplicates
                cte_name = f"{name}_{symbol}".replace("-", "_")
                symbol_ctes.append(
                    f"""
                {cte_name} AS (
                    SELECT *, '{symbol}' as symbol FROM read_parquet('{file_path}')
                )"""
                )

    # --- Macro Data CTEs ---
    macro_ctes = []
    for name, params in macro_endpoints.items():
        file_path = data_dir / (generate_filename(name, params) + ".parquet")
        if file_path.exists():
            # Prefix columns with the endpoint name to avoid collisions
            cols_prefixed = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").fetchall()
            select_list = ", ".join([f'"{col[0]}" as {name}_{col[0]}' for col in cols_prefixed])

            macro_ctes.append(
                f"""
            {name} AS (
                SELECT {select_list} FROM read_parquet('{file_path}')
            )"""
            )

    # --- Main Query Construction ---
    query = "WITH "

    # Add all CTEs
    all_ctes = symbol_ctes + macro_ctes
    query += ",\n".join(all_ctes)

    # Combine all symbol data
    # This assumes a common 'date' column exists after being processed by fetch_data_df
    # and that the data is daily or can be aggregated to daily.
    # For simplicity, we'll focus on TIME_SERIES_DAILY as the base for joining.
    
    # Create a base of all dates and symbols
    all_symbol_dates_parts = []
    for symbol in STOCK_SYMBOLS:
        p = ENDPOINTS["TIME_SERIES_DAILY"].copy()
        p["symbol"] = symbol
        file_path = data_dir / (generate_filename("TIME_SERIES_DAILY", p) + ".parquet")
        if file_path.exists():
            all_symbol_dates_parts.append(f"SELECT date, '{symbol}' as symbol FROM read_parquet('{file_path}')")
    
    all_symbol_dates_query = " UNION ALL ".join(all_symbol_dates_parts)

    query += f"""
    , all_symbol_dates AS (
        {all_symbol_dates_query}
    )
    , combined_symbols AS (
        SELECT *
        FROM all_symbol_dates
    """
    # Left join all other symbol data onto the base time series
    for symbol in STOCK_SYMBOLS:
        for name in symbol_endpoints:
            if name != "TIME_SERIES_DAILY":
                 p = symbol_endpoints[name].copy()
                 p["symbol"] = symbol
                 file_path = data_dir / (generate_filename(name, p) + ".parquet")
                 if file_path.exists():
                    query += f"""
                    LEFT JOIN {name}_{symbol} ON all_symbol_dates.date = {name}_{symbol}.date AND all_symbol_dates.symbol = {name}_{symbol}.symbol
                    """
    query += ")"

    # ASOF join all macro data
    query += """
    , final_dataset AS (
        SELECT
            cs.*,
    """
    macro_cols = []
    for name in macro_endpoints:
        file_path = data_dir / (generate_filename(name, macro_endpoints[name]) + ".parquet")
        if file_path.exists():
            cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").fetchall()
            macro_cols.extend([f'{name}_{col[0]}' for col in cols if col[0] != 'date'])

    query += ",\n".join(macro_cols)
    query += """
        FROM combined_symbols cs
    """
    
    # Chain ASOF joins for all macro tables
    for i, name in enumerate(macro_endpoints):
        file_path = data_dir / (generate_filename(name, macro_endpoints[name]) + ".parquet")
        if file_path.exists():
            join_alias = f"m{i}"
            date_col_prefixed = f"{name}_date"
            query += f"""
            ASOF LEFT JOIN {name} {join_alias} ON cs.date >= {join_alias}.{date_col_prefixed}
            """
    query += """
    )
    SELECT * FROM final_dataset ORDER BY date DESC, symbol;
    """
    con.close()
    final_dataset = get_dataset(query)

    if not final_dataset.empty:
        logger.info("\\nSuccessfully created dataset with DuckDB.")
        output_path = data_dir / "final_dataset.parquet"
        final_dataset.to_parquet(output_path)
        logger.info(f"Saved final dataset to {output_path}")
        logger.info("\\nDataset Info:")
        final_dataset.info()
        logger.info("\\nDataset Head:")
        logger.info(final_dataset.head())
    else:
        logger.error("\\nCould not generate the dataset with DuckDB.")

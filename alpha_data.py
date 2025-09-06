import os
import itertools
from pathlib import Path
from functools import wraps
from settings import settings
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import alpha_vantage_api as ava

load_dotenv()

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
    symbol_like_keys = ['symbol', 'keywords', 'tickers', 'from_symbol', 'to_symbol']
    symbol_value = None
    for key in symbol_like_keys:
        if key in params_copy:
            symbol_value = params_copy.pop(key)
            filename_parts.insert(1, f"symbol_{symbol_value}")
            break

    # Add remaining static parameters
    for key, value in sorted(params_copy.items()):
        if key != 'apikey' and isinstance(value, str):
            filename_parts.append(f"{key}_{value}")

    return "_".join(filename_parts)

def cache_to_parquet(data_dir):
    """
    Decorator to cache the output of a function to a Parquet file.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(endpoint_name, params, *args, **kwargs):
            filepath = Path(data_dir) / (generate_filename(endpoint_name, params) + ".parquet")
            
            if filepath.exists():
                print(f"Loading from cache: {filepath}")
                return pd.read_parquet(filepath)

            df = func(endpoint_name, params, *args, **kwargs)

            if not df.empty:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(filepath)
                print(f"Saved to cache: {filepath}")
            
            return df
        return wrapper
    return decorator

def fetch_data(endpoint_name, params) -> dict | str | None:
    """Fetches data from a single Alpha Vantage endpoint."""
    full_params = {"function": endpoint_name, "apikey": os.environ["ALPHA_VANTAGE_API_KEY"], **params}
    print(f"Fetching data for {endpoint_name} with params: {params}")
    try:
        response = requests.get(ava.BASE_URL, params=full_params)
        response.raise_for_status() # Raise an exception for bad status codes
        # Alpha Vantage sometimes returns an error message in JSON even for 200 status
        if 'Error Message' in response.text or 'Note' in response.text:
             print(f"API returned an error or note for {endpoint_name}: {response.text}")
             return None
        
        if params.get("datatype", "json") == "json":
            return response.json()
        else:
            return response.text
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {endpoint_name}: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data returned for {endpoint_name}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {endpoint_name}: {e}")
        return None

@cache_to_parquet(settings['data_dir'])
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
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        except Exception as e:
            print(f"Error parsing CSV data for {endpoint_name}: {e}")
            df = pd.DataFrame()

    # Handle JSON data
    elif isinstance(data, dict):
        try:
            # Handle time series data
            if any('Time Series' in k for k in data.keys()):
                time_series_key = next(k for k in data.keys() if 'Time Series' in k)
                time_series_data = data[time_series_key]
                if isinstance(time_series_data, dict):
                    df = pd.DataFrame.from_dict(time_series_data, orient='index')
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df.index = pd.to_datetime(df.index)
                
            # Handle fundamental data (quarterly/annual reports)
            elif any(k in data for k in ['quarterlyReports', 'annualReports']):
                reports_key = 'quarterlyReports' if 'quarterlyReports' in data else 'annualReports'
                if isinstance(data.get(reports_key), list):
                    df = pd.DataFrame(data[reports_key])
                    if 'fiscalDateEnding' in df.columns:
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                    numeric_cols = df.select_dtypes(include=['object']).columns
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Handle other JSON structures
            else:
                # Attempt to build a DataFrame, assuming a simple structure
                df = pd.DataFrame.from_dict(data, orient='index').transpose()
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)

        except Exception as e:
            print(f"Error parsing JSON data for {endpoint_name}: {e}")
            df = pd.DataFrame()
    
    return df

def get_timeseries_data(endpoints: dict, stock_symbols: list, save: bool = True) -> pd.DataFrame:
    """
    Fetches data for multiple endpoints, separates symbol-specific from macro data,
    and constructs a final dataset for machine learning.
    """
    macro_endpoints = {}
    symbol_endpoints = {}

    for name, params in endpoints.items():
        if any(key in params for key in ['symbol', 'symbols', 'tickers']):
            symbol_endpoints[name] = params
        else:
            macro_endpoints[name] = params

    # 1. Fetch and combine all macro data
    macro_dfs = []
    for name, params in macro_endpoints.items():
        df = fetch_data_df(name, params, save)
        if not df.empty:
            # Add prefix to avoid column name collisions
            df = df.add_prefix(f"{name}_")
            macro_dfs.append(df)
    
    macro_data = pd.DataFrame()
    if macro_dfs:
        macro_data = pd.concat(macro_dfs, axis=1)
        # Forward-fill macro data to align with daily stock data
        macro_data = macro_data.ffill()

    # 2. Fetch data for each symbol and combine with macro data
    all_symbol_data = []
    for symbol in stock_symbols:
        symbol_dfs = []
        for name, params in symbol_endpoints.items():
            # Create a copy of params and update the symbol
            p = params.copy()
            p['symbol'] = symbol
            
            df = fetch_data_df(name, p, save)
            if not df.empty:
                # Add prefix for clarity
                df = df.add_prefix(f"{name}_")
                symbol_dfs.append(df)
        
        if not symbol_dfs:
            continue

        # Combine all data for the current symbol
        combined_symbol_df = pd.concat(symbol_dfs, axis=1)
        combined_symbol_df['symbol'] = symbol
        all_symbol_data.append(combined_symbol_df)

    if not all_symbol_data:
        return pd.DataFrame()

    # 3. Combine all symbols into one large DataFrame
    final_df = pd.concat(all_symbol_data, axis=0, sort=True)
    
    # Reset index to bring datetime into a column, then set multi-index
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'date'}, inplace=True)
    
    # 4. Join with macro data
    if not macro_data.empty:
        # Use merge_asof for precise point-in-time joining
        final_df = final_df.sort_values('date')
        macro_data = macro_data.sort_index()
        final_df = pd.merge_asof(final_df, macro_data, left_on='date', right_index=True, direction='backward')

    # Set final index
    final_df.set_index(['date', 'symbol'], inplace=True)
    
    return final_df

# Read stock symbols from a file
try:
    with open("stocks.txt", "r") as f:
        STOCK_SYMBOLS = sorted(list(set([line.strip() for line in f if line.strip()])))
except FileNotFoundError:
    print("Error: stocks.txt not found. Using default symbols.")
    STOCK_SYMBOLS = ["IBM", "AAPL"]

# Define non-premium endpoints and their parameters
ENDPOINTS = {
    "TIME_SERIES_DAILY": {"symbol": STOCK_SYMBOLS, "outputsize": "full", "datatype": "csv"},
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
    data_dir = Path(settings['data_dir'])
    data_dir.mkdir(exist_ok=True)

    # Fetch all data and create the final dataset
    ml_dataset = get_timeseries_data(ENDPOINTS, STOCK_SYMBOLS, save=True)

    if not ml_dataset.empty:
        # Save the final combined dataset
        output_path = data_dir / "ml_dataset.parquet"
        ml_dataset.to_parquet(output_path)
        print(f"\nSuccessfully created and saved the final dataset to {output_path}")
        print("\nDataset Info:")
        ml_dataset.info()
        print("\nDataset Head:")
        print(ml_dataset.head())
    else:
        print("\nCould not generate the dataset.")
import os
import requests
import pandas as pd
from io import StringIO

# Replace with your actual Alpha Vantage API key
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://www.alphavantage.co/query"

# Read stock symbols from stocks.txt
try:
    with open("c:/Users/roman/Projects/alpha_vantage/stocks.txt", "r") as f:
        STOCK_SYMBOLS = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: stocks.txt not found.")
    STOCK_SYMBOLS = []

# Define some common crypto symbols and forex pairs
CRYPTO_SYMBOLS = ["BTC", "ETH", "DOGE"]
CRYPTO_MARKETS = ["USD", "EUR"]
FOREX_PAIRS = [("USD", "JPY"), ("EUR", "USD"), ("GBP", "USD")]

# Define non-premium endpoints and their parameters based on user requirements
# Prioritize daily interval, use lowest available non-intraday if daily isn't an option
# Use full outputsize where applicable, and csv datatype
ENDPOINTS = {
    "TIME_SERIES_DAILY": {"symbol": STOCK_SYMBOLS, "outputsize": "full", "datatype": "csv"},
    "TIME_SERIES_WEEKLY": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    "TIME_SERIES_MONTHLY": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    "GLOBAL_QUOTE": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    "SYMBOL_SEARCH": {"keywords": STOCK_SYMBOLS, "datatype": "csv"}, # Will call for each keyword
    "MARKET_STATUS": {},
    "HISTORICAL_OPTIONS": {"symbol": STOCK_SYMBOLS, "date": "2024-05-20", "datatype": "csv"}, # Example date
    "NEWS_SENTIMENT": {"tickers": ",".join(STOCK_SYMBOLS + [f"CRYPTO:{s}" for s in CRYPTO_SYMBOLS] + [f"FOREX:{p[0]}{p[1]}" for p in FOREX_PAIRS]), "limit": 1000, "datatype": "csv"}, # Combine all symbols
    "EARNINGS_CALL_TRANSCRIPT": {"symbol": STOCK_SYMBOLS, "quarter": "2024Q1"}, # Example quarter
    "TOP_GAINERS_LOSERS": {},
    "INSIDER_TRANSACTIONS": {"symbol": STOCK_SYMBOLS},
    "OVERVIEW": {"symbol": STOCK_SYMBOLS},
    "ETF_PROFILE": {"symbol": ["SPY"]}, # Assuming SPY is in stocks.txt or is a relevant ETF
    "DIVIDENDS": {"symbol": STOCK_SYMBOLS},
    "SPLITS": {"symbol": STOCK_SYMBOLS},
    "INCOME_STATEMENT": {"symbol": STOCK_SYMBOLS},
    "BALANCE_SHEET": {"symbol": STOCK_SYMBOLS},
    "CASH_FLOW": {"symbol": STOCK_SYMBOLS},
    "EARNINGS": {"symbol": STOCK_SYMBOLS},
    "LISTING_STATUS": {"state": "active", "datatype": "csv"},
    "EARNINGS_CALENDAR": {"horizon": "3month", "datatype": "csv"}, # Can add symbol here too
    "IPO_CALENDAR": {"datatype": "csv"},
    "CURRENCY_EXCHANGE_RATE": {"pairs": FOREX_PAIRS + [(c, m) for c in CRYPTO_SYMBOLS for m in CRYPTO_MARKETS]}, # Will call for each pair
    "FX_DAILY": {"pairs": FOREX_PAIRS, "outputsize": "full", "datatype": "csv"}, # Will call for each pair
    "FX_WEEKLY": {"pairs": FOREX_PAIRS, "datatype": "csv"}, # Will call for each pair
    "FX_MONTHLY": {"pairs": FOREX_PAIRS, "datatype": "csv"}, # Will call for each pair
    "DIGITAL_CURRENCY_DAILY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
    "DIGITAL_CURRENCY_WEEKLY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
    "DIGITAL_CURRENCY_MONTHLY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
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
    "TREASURY_YIELD": {"interval": "daily", "maturity": ["10year"]}, # Example maturity
    "FEDERAL_FUNDS_RATE": {"interval": "daily", "datatype": "csv"},
    "CPI": {"interval": "monthly", "datatype": "csv"},
    "INFLATION": {"datatype": "csv"},
    "RETAIL_SALES": {"datatype": "csv"},
    "DURABLES": {"datatype": "csv"},
    "UNEMPLOYMENT": {"datatype": "csv"},
    "NONFARM_PAYROLL": {"datatype": "csv"},
}

# Data structures to store results
time_series_data = {
    "daily": {},
    "weekly": {},
    "monthly": {},
    "quarterly": {},
    "annual": {}
}
fundamental_data = {}
utility_data = {}
commodity_data = {}
economic_indicator_data = {}
forex_data = {}
crypto_data = {}

# Summary table for stocks
stock_summary_table = pd.DataFrame(index=STOCK_SYMBOLS)

def fetch_data(endpoint_name, params):
    """Fetches data from a single Alpha Vantage endpoint."""
    full_params = {"function": endpoint_name, "apikey": API_KEY, **params}
    print(f"Fetching data for {endpoint_name} with params: {full_params}")
    try:
        response = requests.get(BASE_URL, params=full_params)
        response.raise_for_status() # Raise an exception for bad status codes
        # Alpha Vantage sometimes returns an error message in JSON even for 200 status
        if 'Error Message' in response.text or 'Note' in response.text:
             print(f"API returned an error or note for {endpoint_name}: {response.text}")
             return None
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {endpoint_name}: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data returned for {endpoint_name}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {endpoint_name}: {e}")
        return None

# Iterate through endpoints and fetch data
for endpoint_name, params_def in ENDPOINTS.items():
    if "symbol" in params_def:
        for symbol in params_def["symbol"]:
            current_params = {k: v for k, v in params_def.items() if k != "symbol"}
            current_params["symbol"] = symbol
            df = fetch_data(endpoint_name, current_params)
            if df is not None:
                if "TIME_SERIES" in endpoint_name:
                    interval = endpoint_name.split("_")[-1].lower() # daily, weekly, monthly
                    if interval in time_series_data:
                         time_series_data[interval][symbol] = df
                    else: # Handle adjusted time series
                         time_series_data[interval + "_adjusted"][symbol] = df
                elif endpoint_name in ["GLOBAL_QUOTE", "OVERVIEW", "ETF_PROFILE", "DIVIDENDS", "SPLITS", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS", "INSIDER_TRANSACTIONS"]:
                    fundamental_data[f"{endpoint_name}_{symbol}"] = df
                    # Add to stock summary table
                    if endpoint_name == "GLOBAL_QUOTE" and not df.empty:
                         # Assuming the CSV has columns like 'symbol', 'latest_trading_day', 'price', 'volume'
                         if 'symbol' in df.columns and 'price' in df.columns:
                             stock_summary_table.loc[symbol, 'Latest Price'] = df.loc[0, 'price']
                             stock_summary_table.loc[symbol, 'Latest Trading Day'] = df.loc[0, 'latest_trading_day']
                         elif 'Symbol' in df.columns and 'Price' in df.columns: # Handle potential variations in column names
                             stock_summary_table.loc[symbol, 'Latest Price'] = df.loc[0, 'Price']
                             stock_summary_table.loc[symbol, 'Latest Trading Day'] = df.loc[0, 'Latest Trading Day']

                    elif endpoint_name == "OVERVIEW" and not df.empty:
                         # Assuming the CSV has relevant columns like 'MarketCapitalization', 'PERatio'
                         if 'MarketCapitalization' in df.columns:
                             stock_summary_table.loc[symbol, 'Market Cap'] = df.loc[0, 'MarketCapitalization']
                         if 'PERatio' in df.columns:
                             stock_summary_table.loc[symbol, 'PE Ratio'] = df.loc[0, 'PERatio']

                elif endpoint_name == "SYMBOL_SEARCH":
                     utility_data[f"{endpoint_name}_{symbol}"] = df # Store search results per keyword
                elif endpoint_name == "HISTORICAL_OPTIONS":
                     utility_data[f"{endpoint_name}_{symbol}"] = df # Store options data per symbol

    elif "pairs" in params_def:
         for from_c, to_c in params_def["pairs"]:
             current_params = {k: v for k, v in params_def.items() if k != "pairs"}
             current_params["from_currency"] = from_c
             current_params["to_currency"] = to_c
             df = fetch_data(endpoint_name, current_params)
             if df is not None:
                 if "FX_" in endpoint_name:
                     interval = endpoint_name.split("_")[-1].lower()
                     if interval in forex_data:
                         forex_data[interval][f"{from_c}_{to_c}"] = df
                     else:
                         forex_data[interval] = {f"{from_c}_{to_c}": df}
                 elif endpoint_name == "CURRENCY_EXCHANGE_RATE":
                      forex_data[f"{endpoint_name}_{from_c}_{to_c}"] = df

    elif "symbols" in params_def and "markets" in params_def:
         for symbol in params_def["symbols"]:
             for market in params_def["markets"]:
                 current_params = {k: v for k, v in params_def.items() if k not in ["symbols", "markets"]}
                 current_params["symbol"] = symbol
                 current_params["market"] = market
                 df = fetch_data(endpoint_name, current_params)
                 if df is not None:
                     if "DIGITAL_CURRENCY_" in endpoint_name:
                         interval = endpoint_name.split("_")[-1].lower()
                         if interval in crypto_data:
                             crypto_data[interval][f"{symbol}_{market}"] = df
                         else:
                             crypto_data[interval] = {f"{symbol}_{market}": df}

    elif "maturity" in params_def:
         for maturity in params_def["maturity"]:
             current_params = {k: v for k, v in params_def.items() if k != "maturity"}
             current_params["maturity"] = maturity
             df = fetch_data(endpoint_name, current_params)
             if df is not None:
                 economic_indicator_data[f"{endpoint_name}_{maturity}"] = df

    elif endpoint_name in ["WTI", "BRENT", "NATURAL_GAS", "COPPER", "ALUMINUM", "WHEAT", "CORN", "COTTON", "SUGAR", "COFFEE", "ALL_COMMODITIES"]:
         df = fetch_data(endpoint_name, params_def)
         if df is not None:
             commodity_data[endpoint_name] = df

    elif endpoint_name in ["REAL_GDP", "REAL_GDP_PER_CAPITA", "FEDERAL_FUNDS_RATE", "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"]:
         df = fetch_data(endpoint_name, params_def)
         if df is not None:
             economic_indicator_data[endpoint_name] = df

    elif endpoint_name in ["MARKET_STATUS", "TOP_GAINERS_LOSERS", "LISTING_STATUS", "EARNINGS_CALENDAR", "IPO_CALENDAR", "NEWS_SENTIMENT"]:
         df = fetch_data(endpoint_name, params_def)
         if df is not None:
             utility_data[endpoint_name] = df

# Print summaries or save dataframes
print("\n--- Data Fetching Complete ---")

print("\n--- Stock Summary Table ---")
print(stock_summary_table)

print("\n--- Time Series Data (Daily) ---")
for symbol, df in time_series_data["daily"].items():
    print(f"\n{symbol} Daily Data:")
    print(df.head())

# You can similarly print or process other data structures
# print("\n--- Fundamental Data ---")
# for name, df in fundamental_data.items():
#     print(f"\n{name}:")
#     print(df.head())

# print("\n--- Utility Data ---")
# for name, df in utility_data.items():
#     print(f"\n{name}:")
#     print(df.head())

# ... and so on for other data types

# Example of how to access a specific DataFrame:
# if "AAPL" in time_series_data["daily"]:
#     aapl_daily_df = time_series_data["daily"]["AAPL"]
#     print("\nAAPL Daily Data (from dictionary):")
#     print(aapl_daily_df.head())

# You can save these DataFrames to files (e.g., CSV, Parquet) if needed
# For example:
# for symbol, df in time_series_data["daily"].items():
#     df.to_csv(f"{symbol}_daily.csv")

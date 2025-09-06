import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import alpha_vantage_api as ava
from urllib.parse import urlparse, parse_qs

# Replace with your actual Alpha Vantage API key
load_dotenv()

# Read stock symbols from stocks.txt
try:
    with open("stocks.txt", "r") as f:
        STOCK_SYMBOLS = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: stocks.txt not found.")
    STOCK_SYMBOLS = ["IBM"]

# Define non-premium endpoints and their parameters based on user requirements
# Prioritize daily interval, use lowest available non-intraday if daily isn't an option
# Use full outputsize where applicable, and csv datatype
ENDPOINTS = {
    "TIME_SERIES_DAILY": {"symbol": STOCK_SYMBOLS, "outputsize": "full", "datatype": "csv"},
    # "TIME_SERIES_WEEKLY": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    # "TIME_SERIES_MONTHLY": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    # "GLOBAL_QUOTE": {"symbol": STOCK_SYMBOLS, "datatype": "csv"},
    # "SYMBOL_SEARCH": {"keywords": STOCK_SYMBOLS, "datatype": "csv"}, # Will call for each keyword
    # "MARKET_STATUS": {},
    # "HISTORICAL_OPTIONS": {"symbol": STOCK_SYMBOLS, "date": "2024-05-20", "datatype": "csv"}, # Example date
    # "NEWS_SENTIMENT": {"tickers": ",".join(STOCK_SYMBOLS + [f"CRYPTO:{s}" for s in CRYPTO_SYMBOLS] + [f"FOREX:{p[0]}{p[1]}" for p in FOREX_PAIRS]), "limit": 1000, "datatype": "csv"}, # Combine all symbols
    # "EARNINGS_CALL_TRANSCRIPT": {"symbol": STOCK_SYMBOLS, "quarter": "2024Q1"}, # Example quarter
    # "TOP_GAINERS_LOSERS": {},
    "INSIDER_TRANSACTIONS": {"symbol": STOCK_SYMBOLS},
    # "OVERVIEW": {"symbol": STOCK_SYMBOLS},
    # "ETF_PROFILE": {"symbol": ["SPY"]}, # Assuming SPY is in stocks.txt or is a relevant ETF
    "DIVIDENDS": {"symbol": STOCK_SYMBOLS},
    # "SPLITS": {"symbol": STOCK_SYMBOLS},
    "INCOME_STATEMENT": {"symbol": STOCK_SYMBOLS},
    "BALANCE_SHEET": {"symbol": STOCK_SYMBOLS},
    "CASH_FLOW": {"symbol": STOCK_SYMBOLS},
    "EARNINGS": {"symbol": STOCK_SYMBOLS},
    # "LISTING_STATUS": {"state": "active", "datatype": "csv"},
    # "EARNINGS_CALENDAR": {"horizon": "3month", "datatype": "csv"}, # Can add symbol here too
    # "IPO_CALENDAR": {"datatype": "csv"},
    # "CURRENCY_EXCHANGE_RATE": {"pairs": FOREX_PAIRS + [(c, m) for c in CRYPTO_SYMBOLS for m in CRYPTO_MARKETS]}, # Will call for each pair
    # "FX_DAILY": {"pairs": FOREX_PAIRS, "outputsize": "full", "datatype": "csv"}, # Will call for each pair
    # "FX_WEEKLY": {"pairs": FOREX_PAIRS, "datatype": "csv"}, # Will call for each pair
    # "FX_MONTHLY": {"pairs": FOREX_PAIRS, "datatype": "csv"}, # Will call for each pair
    # "DIGITAL_CURRENCY_DAILY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
    # "DIGITAL_CURRENCY_WEEKLY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
    # "DIGITAL_CURRENCY_MONTHLY": {"symbols": CRYPTO_SYMBOLS, "markets": CRYPTO_MARKETS, "datatype": "csv"}, # Will call for each symbol/market combo
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

def fetch_data(endpoint_name, params) -> dict | str | None:
    """Fetches data from a single Alpha Vantage endpoint."""
    full_params = {"function": endpoint_name, "apikey": os.environ["ALPHA_VANTAGE_API_KEY"], **params}
    print(f"Fetching data for {endpoint_name} with params: {full_params}")
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

def get_data_df(endpoint_name, params) -> pd.DataFrame:
    """Fetches DataFrame from a single Alpha Vantage endpoint."""
    data = fetch_data(endpoint_name, params)
    # This is where the endpoint-specific parsing logic would go
    return pd.DataFrame(data)

def fetch_example_endpoints(endpoint_outputs_dir: str = "endpoint_outputs"):
    """
    Fetches data for all example endpoints found in the documentation
    and saves the raw output to the endpoint_outputs_dir folder.
    """
    # Make sure the output directory exists
    if not os.path.exists(endpoint_outputs_dir):
        os.makedirs(endpoint_outputs_dir)

    example_urls = [
        'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&outputsize=full&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=IBM&outputsize=full&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=demo',
        'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=IBM&quarter=2024Q1&apikey=demo',
        'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey=demo',
        'https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol=SPY&apikey=demo',
        'https://www.alphavantage.co/query?function=DIVIDENDS&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=SPLITS&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=EARNINGS&symbol=IBM&apikey=demo',
        'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=JPY&apikey=demo',
        'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&outputsize=full&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol=EUR&to_symbol=USD&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol=EUR&to_symbol=USD&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_WEEKLY&symbol=BTC&market=USD&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_MONTHLY&symbol=BTC&market=USD&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=WTI&interval=daily&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=NATURAL_GAS&interval=daily&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=COPPER&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=ALUMINUM&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=WHEAT&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=CORN&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=COTTON&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=SUGAR&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=COFFEE&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=ALL_COMMODITIES&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=INFLATION&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=DURABLES&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=demo&datatype=csv',
        'https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey=demo&datatype=csv',
    ]

    for url in example_urls:
        try:
            # Extract params from URL
            url_params = urlparse(url)
            query = parse_qs(url_params.query)
            
            endpoint_name = query.get("function", [None])[0]
            if not endpoint_name:
                print(f"Skipping URL due to missing function: {url}")
                continue

            # Construct filename
            params_for_filename = {k: v[0] for k, v in query.items() if k not in ['function', 'apikey']}
            filename_parts = [endpoint_name]
            for key, value in sorted(params_for_filename.items()):
                filename_parts.append(f"{key}_{value}")
            
            file_ext = "csv" if query.get("datatype", ["json"])[0] == "csv" else "json"
            filename = f"{'_'.join(filename_parts)}.{file_ext}"
            filepath = os.path.join(endpoint_outputs_dir, filename)

            # Fetch and save data
            print(f"Fetching example for {endpoint_name} from {url}")
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "w") as f:
                f.write(response.text)
            print(f"Saved to {filepath}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {url}: {e}")

# This is where you should call endpoints in the
# documentation using the demo api key and
fetch_example_endpoints()

import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

def fetch_example_endpoints(endpoint_outputs_dir: str = "endpoint_outputs"):
    """
    Fetches data for all example endpoints found in the documentation
    and saves the raw output to the endpoint_outputs_dir folder.
    """
    # Make sure the output directory exists
    if not os.path.exists(endpoint_outputs_dir):
        os.makedirs(endpoint_outputs_dir)

    example_urls = [
        f'https://www.alphavantage.co/query?function=WHEAT&interval=monthly&datatype=csv&apikey={os.environ["ALPHA_VANTAGE_API_KEY"]}',
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
        'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo',
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

    for url in example_urls[:1]:
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

if __name__ == "__main__":
    load_dotenv()
    fetch_example_endpoints()
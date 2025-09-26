from alpha_vantage import fetch_alpha_vantage_data
from utils import read_stock_symbols, get_default_endpoints

if __name__ == "__main__":
    # Read stock symbols from file
    stock_symbols = read_stock_symbols()

    # Dynamically create endpoints from the schema
    endpoints = get_default_endpoints()

    # Fetch the data
    df = fetch_alpha_vantage_data(
        symbols=stock_symbols, 
        endpoints=endpoints, 
        force_refresh=False
    )

    # Print the first few rows of the combined DataFrame
    if not df.empty:
        print("Successfully fetched and combined data:")
        print(df.head())
    else:
        print("Could not fetch any data.")


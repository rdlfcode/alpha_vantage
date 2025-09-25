from alpha_vantage import fetch_alpha_vantage_data
from utils import read_stock_symbols
from alpha_vantage_schema import ALPHA_VANTAGE_SCHEMA

def get_default_params(endpoint_name: str) -> dict:
    """
    Get default parameters for a given endpoint from the schema.
    This function simplifies parameter creation by extracting the first valid
    value for parameters that require a specific choice (e.g., interval).
    """
    params = {}
    if endpoint_name in ALPHA_VANTAGE_SCHEMA:
        for param, values in ALPHA_VANTAGE_SCHEMA[endpoint_name].items():
            if isinstance(values, list):
                # Use the first value as the default for list-based params
                params[param] = values[0]
            elif values == "string":
                # Skip string parameters like 'symbol' which are handled separately
                continue
            else:
                params[param] = values
    return params

if __name__ == "__main__":
    # Read stock symbols from file
    stock_symbols = read_stock_symbols()

    # Dynamically create endpoints from the schema
    endpoints = {name: get_default_params(name) for name in ALPHA_VANTAGE_SCHEMA}

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


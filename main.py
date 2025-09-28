from dotenv import load_dotenv
from alpha_vantage import AlphaVantageClient
from utils import get_default_endpoints

if __name__ == "__main__":
    load_dotenv()

    # Create AV client (for reuse later)
    client = AlphaVantageClient()

    # Dynamically create endpoints from the schema
    endpoints = get_default_endpoints()

    # Fetch the data
    df = client.get_data(
        endpoints=endpoints,
        force_refresh=True
    )

    # Print the first few rows of the combined DataFrame
    if not df.empty:
        print("Successfully fetched and combined data:")
        print(df.head())
    else:
        print("Could not fetch any data.")
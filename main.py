from dotenv import load_dotenv
from alpha_vantage import AlphaVantageClient
import alpha_vantage_schema as avs

if __name__ == "__main__":
   import duckdb
   from pathlib import Path
   from settings import settings

   conn = duckdb.connect(Path(settings.get("db_path", "data/alpha_vantage.db")))
   result = conn.execute("SELECT * FROM TIME_SERIES_DAILY WHERE symbol = 'GOOG'").df()
   print(result)
   # load_dotenv()

   # # Create AV client (for reuse later)
   # client = AlphaVantageClient()

   # # Dynamically create endpoints from the schema
   # endpoints = avs.DEFAULT_ENDPOINTS

   # # Fetch the data
   # df = client.get_data(
   #    symbols=["GOOG"],
   #    endpoints=endpoints,
   #    force_refresh=False
   # )

   # # Print the first few rows of the combined DataFrame
   # if not df.empty:
   #    print("Successfully fetched and combined data:")    
   #    print(df.head())
   # else:
   #    print("Could not fetch any data.")

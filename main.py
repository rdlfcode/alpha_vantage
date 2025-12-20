import duckdb
import logging
from settings import settings
from alpha_vantage import AlphaVantageClient
from alpha_vantage_schema import ENDPOINT_TO_TABLE_MAP, TABLE_SCHEMAS, DEFAULT_ENDPOINTS

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicates/console pollution
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Add FileHandler
file_handler = logging.FileHandler("av.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Alpha Vantage Data Updater...")

    # Ensure tables exist
    conn = duckdb.connect(settings.get("db_path"))
    tables = set()
    for endpoint_name in DEFAULT_ENDPOINTS:
        tables.add(ENDPOINT_TO_TABLE_MAP.get(endpoint_name, endpoint_name).upper())

    for table_name in tables:
        schema_sql = TABLE_SCHEMAS.get(table_name)
        if schema_sql:
            create_sql = schema_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
            conn.execute(create_sql)

    client = AlphaVantageClient(db_conn=conn)

    # Update data
    df = client.get_data()

if __name__ == "__main__":
   main()

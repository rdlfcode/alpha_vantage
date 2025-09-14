import os
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from settings import settings
import pandas as pd
from dotenv import load_dotenv
import duckdb
import logging

import alpha_vantage_api as ava
from alpha_vantage_client import AlphaVantageClient

load_dotenv()

# --- Logging Setup ---
log_settings = settings.get("logging", {})
logging.basicConfig(
    level=log_settings.get("level", "INFO"),
    format=log_settings.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)

def read_stock_symbols(file_path: str = "stocks.txt") -> List[str]:
    """Read stock symbols from a file, one per line."""
    try:
        with open(file_path, "r") as f:
            return sorted(list(set([line.strip() for line in f if line.strip()])))
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Using default symbols.")
        return ["IBM", "AAPL"]

def get_dataset(sql_query: str, db_path: str) -> pd.DataFrame:
    """
    Executes a SQL query using DuckDB on the cached Parquet files and returns a DataFrame.
    """
    try:
        con = duckdb.connect(database=db_path, read_only=False)
        logger.info(f"Executing query:\n{sql_query}")
        result_df = con.execute(sql_query).fetchdf()
        con.close()
        return result_df
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}")
        return pd.DataFrame()

def build_combined_dataset(client: AlphaVantageClient, symbols: List[str], endpoints: Dict[str, Dict],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Build a combined dataset from downloaded data files with optional date filtering.
    """
    ctes = []
    base_tables = []

    # Process symbol-specific data
    for symbol in symbols:
        for name in [ep for ep in endpoints if ep in ava.SYMBOL_ENDPOINTS]:
            params = endpoints[name].copy()
            params["symbol"] = symbol
            file_path = client.data_dir / (client._generate_filename(name, params) + ".parquet")
            if file_path.exists():
                table_name = f"{name}_{symbol}".lower()
                cte = f"""
                    {table_name} AS (
                        SELECT *, '{symbol}' as symbol
                        FROM read_parquet('{file_path}')
                        WHERE 1=1
                        {f"AND date >= '{start_date}'" if start_date else ""}
                        {f"AND date <= '{end_date}'" if end_date else ""}
                    )
                """
                ctes.append(cte.strip())
                if name == "TIME_SERIES_DAILY":
                    base_tables.append(table_name)

    # Process macro endpoints
    for name in [ep for ep in endpoints if ep in ava.MACRO_ENDPOINTS]:
        params = endpoints[name]
        file_path = client.data_dir / (client._generate_filename(name, params) + ".parquet")
        if file_path.exists():
            table_name = name.lower()
            cte = f"""
                {table_name} AS (
                    SELECT *
                    FROM read_parquet('{file_path}')
                    WHERE 1=1
                    {f"AND date >= '{start_date}'" if start_date else ""}
                    {f"AND date <= '{end_date}'" if end_date else ""}
                )
            """
            ctes.append(cte.strip())

    if not base_tables:
        logger.warning("No time series data found for any symbol")
        return pd.DataFrame()

    query = "WITH " + ",\n".join(ctes)
    query += f", combined_base AS (SELECT * FROM {base_tables[0]} {' UNION ALL '.join(f'SELECT * FROM {t}' for t in base_tables[1:])})"
    query += ", enriched_symbols AS (SELECT cb.* FROM combined_base cb"

    for symbol in symbols:
        for name in [ep for ep in endpoints if ep in ava.SYMBOL_ENDPOINTS and ep != "TIME_SERIES_DAILY"]:
            table_name = f"{name}_{symbol}".lower()
            if f"{table_name} AS" in query:
                query += f" LEFT JOIN {table_name} USING (date, symbol)"
    query += ")"

    query += " SELECT es.*"
    macro_joins = []
    for name in [ep for ep in endpoints if ep in ava.MACRO_ENDPOINTS]:
        table_name = name.lower()
        if f"{table_name} AS" in query:
            query += f", {table_name}.*"
            macro_joins.append(f" LEFT JOIN {table_name} ON es.date >= {table_name}.date AND es.date < LEAD({table_name}.date, 1) OVER (ORDER BY {table_name}.date)")

    query += " FROM enriched_symbols es"
    query += " ".join(macro_joins)
    query += " ORDER BY date DESC, symbol;"

    try:
        result_df = get_dataset(query, client.db_path)
        if not result_df.empty:
            output_path = client.data_dir / "final_dataset.parquet"
            result_df.to_parquet(output_path)
        return result_df
    except Exception as e:
        logger.error(f"Error building combined dataset: {e}")
        return pd.DataFrame()

def fetch_alpha_vantage_data(symbols: Optional[List[str]] = None,
                             endpoints: Optional[Dict[str, Dict]] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             force_refresh: bool = False) -> pd.DataFrame:
    """
    Main function to fetch and process Alpha Vantage data.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")

    client = AlphaVantageClient(
        api_key=api_key,
        data_dir=settings["data_dir"],
        db_path=settings["db_path"],
        rpm=settings["AlphaVantageRPM"],
        rpd=settings["AlphaVantageRPD"]
    )

    symbols = symbols or read_stock_symbols()
    endpoints = endpoints or ava.ALPHA_VANTAGE_SCHEMA
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    valid_endpoints = {k: v for k, v in endpoints.items() if k in ava.SYMBOL_ENDPOINTS + ava.MACRO_ENDPOINTS}

    client.download_all_data(valid_endpoints, symbols, force_refresh=force_refresh)

    return build_combined_dataset(client, symbols, valid_endpoints, start_date, end_date)


import pandas as pd
import duckdb
from alpha_vantage import AlphaVantageClient
from alpha_vantage_schema import ALPHA_VANTAGE_SCHEMA
from settings import settings
import time
import json
import os

def get_test_params(endpoint, schema_def, symbol="IBM"):
    params = {}
    
    # Common parameters
    if "symbol" in schema_def:
        params["symbol"] = symbol
    if "symbols" in schema_def:
        params["symbols"] = symbol
    if "keywords" in schema_def:
        params["keywords"] = "Microsoft"
    if "from_currency" in schema_def:
        params["from_currency"] = "USD"
    if "to_currency" in schema_def:
        params["to_currency"] = "EUR"
    if "from_symbol" in schema_def:
        params["from_symbol"] = "EUR"
    if "to_symbol" in schema_def:
        params["to_symbol"] = "USD"
    if "market" in schema_def:
        params["market"] = "CNY"
    if "tickers" in schema_def:
        params["tickers"] = "AAPL"
        
    # Required enums (pick first)
    for key, value in schema_def.items():
        if isinstance(value, list) and value:
            params[key] = value[0]
        elif key not in params and key not in ["datatype", "function"]:
            # If it's a string/int requirement without enum, provide dummy
            if value == "integer":
                params[key] = "10"
            elif value == "string":
                # already handled mostly, but generic fallback
                pass
                
    return params

def test_all_endpoints():
    db_path = settings.get("db_path", "data/alpha_vantage.db")
    print(f"Connecting to database at: {db_path}")
    conn = duckdb.connect(db_path)
    client = AlphaVantageClient(db_conn=conn)
    
    # Prioritize Fundamental Endpoints for this run
    FUNDAMENTAL_KEYS = [
        "OVERVIEW", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", 
        "EARNINGS", "EARNINGS_ESTIMATES", "LISTING_STATUS", "DIVIDENDS", "SPLITS"
    ]
    
    # Sort so Fundamentals come first
    all_endpoints = list(ALPHA_VANTAGE_SCHEMA.keys())
    # Move fundamentals to front
    for k in reversed(FUNDAMENTAL_KEYS):
        if k in all_endpoints:
            all_endpoints.insert(0, all_endpoints.pop(all_endpoints.index(k)))

    print(f"Testing {len(all_endpoints)} endpoints...")

    for endpoint in all_endpoints:
        schema = ALPHA_VANTAGE_SCHEMA[endpoint]
        print(f"\n[{endpoint}] Testing...")
        
        params = get_test_params(endpoint, schema)
        # Ensure we don't accidentally ask for CSV if we want to retain JSON structure for analysis
        params["datatype"] = "json" 
        
        print(f"Params: {params}")
        
        try:
            # 1. Fetch
            raw_data = client._fetch_data(endpoint, params)
            if not raw_data:
                print("  -> No data returned (or limit reached).")
                continue
                
            if isinstance(raw_data, dict):
                # Check for errors/info
                msg = raw_data.get("Information") or raw_data.get("Note") or raw_data.get("Error Message")
                if msg:
                     print(f"  -> API Message: {msg}")
                     if "rate limit" in str(msg).lower():
                         print("!!! Rate Limit Hit. Stopping.")
                         break
                     if "Error Message" in raw_data:
                         continue

            # 2. Parse (using existing logic to see what happens)
            df = client._parse_response(endpoint, raw_data, params)
            
            if not df.empty:
                print(f"  -> Parsed Shape: {df.shape}")
                print(f"  -> Columns: {df.columns.tolist()}")
                # Print first row to see data types/content
                try:
                    print(f"  -> Sample: {df.iloc[0].to_dict()}")
                except:
                    pass
            else:
                 print("  -> Parsed DataFrame is empty.")
                 print(f"  -> Raw Keys: {raw_data.keys() if isinstance(raw_data, dict) else 'Not Dict'}")

            # 3. Analyze for aggregation
            # Check structure of raw_data for fundamental endpoints
            if endpoint in FUNDAMENTAL_KEYS:
                 if isinstance(raw_data, dict):
                     keys = raw_data.keys()
                     print(f"  -> {endpoint} Raw Keys: {list(keys)}")
                     # Check for reports
                     for k in ["annualReports", "quarterlyReports", "annualEarnings", "quarterlyEarnings"]:
                         if k in raw_data:
                             sample_list = raw_data[k]
                             if sample_list:
                                 print(f"     -> {k} Sample Keys: {sample_list[0].keys()}")

        except Exception as e:
            print(f"  -> EXCEPTION: {e}")
            
        time.sleep(2) # Be nice to API

if __name__ == "__main__":
    test_all_endpoints()
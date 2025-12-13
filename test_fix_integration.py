
import sys
import pandas as pd
import warnings
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from alpha_vantage import AlphaVantageClient

# Filter warnings to ensure we catch them
warnings.simplefilter('always', FutureWarning)

def test_parse_response_time_series():
    print("Testing _parse_response with Time Series data...")
    
    # Mock efficient usage of client (no API keys needed for this static method test usually, 
    # but we need instance for _parse_response)
    # Use in-memory DB to avoid file locking issues during test
    client = AlphaVantageClient(api_key="mock_key", db_conn=":memory:")
    
    # Mock data structure matching what consume_data.py likely gets
    # A dict with "Time Series (Daily)" key
    mock_data = {
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "IBM"},
        "Time Series (Daily)": {
            "2023-10-27": {
                "1. open": "140.00",
                "2. high": "142.00",
                "3. low": "139.00",
                "4. close": "141.00",
                "5. volume": "3000000"
            },
            "2023-10-26": {
                "1. open": "139.00",
                "2. high": "141.00",
                "3. low": "138.00",
                "4. close": "140.00",
                "5. volume": "2500000"
            }
        }
    }
    
    params = {"function": "TIME_SERIES_DAILY", "datatype": "json"}
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = client._parse_response("TIME_SERIES_DAILY", mock_data, params)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame index name: {df.index.name}")
        print(f"DataFrame head:\n{df.head()}")
        
        # Check for our specific warning
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        dtype_warnings = [x for x in future_warnings if "Dtype inference" in str(x.message)]
        
        if dtype_warnings:
            print("❌ FAILED: Dtype inference warning caught!")
            for warning in dtype_warnings:
                print(f"Warning: {warning.message}")
        else:
            print("✅ SUCCESS: No Dtype inference warnings caught.")

    # Functional Correctness Checks
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) == 2, "Should have 2 rows"
    assert "open" in df.columns, "Should have renamed columns (removed '1. ')"
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Index should be datetime"
    assert df.index.name == "dt", "Index name should be 'dt'"

if __name__ == "__main__":
    try:
        test_parse_response_time_series()
        print("\nAll integration tests passed.")
    except Exception as e:
        print(f"\n❌ Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

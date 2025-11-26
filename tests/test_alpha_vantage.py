import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from alpha_vantage import AlphaVantageClient
from alpha_vantage_schema import SYMBOL_ENDPOINTS

# Mock data
MOCK_TIME_SERIES_DAILY = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "IBM",
        "3. Last Refreshed": "2023-10-27",
        "4. Output Size": "Compact",
        "5. Time Zone": "US/Eastern"
    },
    "Time Series (Daily)": {
        "2023-10-27": {
            "1. open": "142.50",
            "2. high": "143.00",
            "3. low": "141.00",
            "4. close": "142.00",
            "5. volume": "3000000"
        },
        "2023-10-26": {
            "1. open": "141.00",
            "2. high": "142.00",
            "3. low": "140.00",
            "4. close": "141.50",
            "5. volume": "2500000"
        }
    }
}

@pytest.fixture
def mock_env_api_key(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")

@pytest.fixture
def client(mock_env_api_key, tmp_path):
    # Use a temporary directory for data and db
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "test.db")
    return AlphaVantageClient(data_dir=str(data_dir), db_path=db_path)

def test_init_no_api_key(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        AlphaVantageClient()

def test_fetch_data_success(client):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TIME_SERIES_DAILY
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Test fetching daily data
        df = client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json"}})
        
        assert not df.empty
        assert len(df) == 2
        assert "close" in df.columns
        assert df.index.name == "dt"
        # Check if symbol column is added
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "IBM"

def test_caching(client):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TIME_SERIES_DAILY
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # First call should fetch from API
        client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json", "outputsize": "compact"}})
        assert mock_get.call_count == 1
        
        # Verify file exists
        expected_path = client.data_dir / "files" / "TIME_SERIES_DAILY" / "symbol_IBM_datatype_json_outputsize_compact.parquet"
        assert expected_path.exists()

        # Second call should load from cache (no new API call)
        client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json", "outputsize": "compact"}})
        assert mock_get.call_count == 1

def test_force_refresh(client):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TIME_SERIES_DAILY
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # First call
        client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json"}}, force_refresh=True)
        assert mock_get.call_count == 1
        
        # Second call with force_refresh=True should trigger another API call
        client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json"}}, force_refresh=True)
        assert mock_get.call_count == 2

def test_database_storage(client):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TIME_SERIES_DAILY
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json"}})
        
        # Verify data is in DuckDB
        import duckdb
        conn = duckdb.connect(client.db_path)
        result = conn.execute("SELECT * FROM TIME_SERIES_DAILY").df()
        conn.close()
        
        assert not result.empty
        assert len(result) == 2
        assert "symbol" in result.columns
        assert result["symbol"].iloc[0] == "IBM"

def test_fetch_data_api_error(client):
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        # Simulate API error response (e.g., rate limit or invalid key)
        mock_response.text = '{"Information": "Thank you for using Alpha Vantage..."}'
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        df = client.get_data(symbols=["IBM"], endpoints={"TIME_SERIES_DAILY": {"datatype": "json"}})
        assert df.empty


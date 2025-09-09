import os
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, List
from dotenv import load_dotenv
import logging

from rate_limiter import RateLimiter
import alpha_vantage_api as ava

load_dotenv()

class AlphaVantageClient:
    """
    A client for interacting with the Alpha Vantage API.
    """

    def __init__(self, api_key: str, data_dir: str = "data", db_path: str = "data/alpha_vantage.db", rpm: int = 75, rpd: int = 25):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.rate_limiter = RateLimiter(requests_per_minute=rpm, requests_per_day=rpd)
        self.logger = logging.getLogger(__name__)

    def _generate_filename(self, endpoint_name: str, params: Dict) -> str:
        """
        Generates a consistent filename from the endpoint and its parameters.
        """
        filename_parts = [endpoint_name]
        params_copy = params.copy()

        symbol_like_keys = ["symbol", "keywords", "tickers", "from_symbol", "to_symbol"]
        for key in symbol_like_keys:
            if key in params_copy:
                filename_parts.insert(1, f"symbol_{params_copy.pop(key)}")
                break

        for key, value in sorted(params_copy.items()):
            if key != "apikey" and isinstance(value, str):
                filename_parts.append(f"{key}_{value}")

        return "_".join(filename_parts)

    def _fetch_data(self, endpoint_name: str, params: Dict) -> Optional[Dict | str]:
        """
        Fetches data from a single Alpha Vantage endpoint.
        """
        self.rate_limiter.wait_if_needed()
        full_params = {
            "function": endpoint_name,
            "apikey": self.api_key,
            **params,
        }
        self.logger.info(f"Fetching data for {endpoint_name} with params: {params}")
        try:
            response = requests.get(ava.BASE_URL, params=full_params)
            response.raise_for_status()
            if "Information" in response.text or "Note" in response.text:
                self.logger.warning(f"API returned an error or note for {endpoint_name}: {response.text}")
                return None
            return response.json() if params.get("datatype", "json") == "json" else response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data for {endpoint_name}: {e}")
            return None

    def _parse_response(self, endpoint_name: str, data: Dict | str, params: Dict) -> pd.DataFrame:
        """
        Parses the raw API response into a pandas DataFrame.
        """
        if data is None:
            return pd.DataFrame()

        df = pd.DataFrame()
        try:
            if params.get("datatype", "json") == "csv" and isinstance(data, str):
                df = pd.read_csv(StringIO(data))
                if "timestamp" in df.columns:
                    df.rename(columns={"timestamp": "date"}, inplace=True)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
            elif isinstance(data, dict):
                # This part can be complex and may need more granular parsing based on endpoint
                # For now, a simplified version of the logic from alpha_data.py
                if any("data" in k for k in data.keys()):
                    time_series_key = next(k for k in data.keys() if "data" in k)
                    time_series_data = data[time_series_key]
                    if isinstance(time_series_data, dict):
                        df = pd.DataFrame.from_dict(time_series_data, orient="index")
                        df = df.apply(pd.to_numeric, errors="coerce")
                elif any(k in data for k in ["quarterlyReports", "annualReports"]):
                    reports_key = "quarterlyReports" if "quarterrepor" in data else "annualReports"
                    df = pd.DataFrame(data[reports_key])
                    if "fiscalDateEnding" in df.columns:
                        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
                        df.set_index("fiscalDateEnding", inplace=True)
                else:
                    df = pd.DataFrame.from_dict(data, orient="index").transpose()
        except Exception as e:
            self.logger.error(f"Error parsing data for {endpoint_name}: {e}")
            return pd.DataFrame()
        return df

    def get_data(self, endpoint_name: str, params: Dict) -> pd.DataFrame:
        """
        Fetches, parses, and caches data for a given endpoint and parameters.
        """
        filepath = self.data_dir / (self._generate_filename(endpoint_name, params) + ".parquet")
        if filepath.exists():
            self.logger.info(f"Loading from cache: {filepath}")
            return pd.read_parquet(filepath)

        raw_data = self._fetch_data(endpoint_name, params)
        df = self._parse_response(endpoint_name, raw_data, params)

        if not df.empty:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath)
            self.logger.info(f"Saved to cache: {filepath}")

        return df

    def download_all_data(self, endpoints: Dict, stock_symbols: List[str], force_refresh: bool = False):
        """
        Iterates through all endpoints and symbols, downloading data.
        """
        self.data_dir.mkdir(exist_ok=True)

        macro_endpoints = {n: p for n, p in endpoints.items() if "symbol" not in p}
        symbol_endpoints = {n: p for n, p in endpoints.items() if "symbol" in p}

        for name, params in macro_endpoints.items():
            filepath = self.data_dir / (self._generate_filename(name, params) + ".parquet")
            if not filepath.exists() or force_refresh:
                self.logger.info(f"Fetching '{name}'...")
                self.get_data(name, params)
            else:
                self.logger.info(f"'{name}' data already cached at {filepath}. Skipping download.")

        for symbol in stock_symbols:
            for name, params in symbol_endpoints.items():
                p = params.copy()
                p["symbol"] = symbol
                filepath = self.data_dir / (self._generate_filename(name, p) + ".parquet")
                if not filepath.exists() or force_refresh:
                    self.logger.info(f"Fetching '{name}' for symbol '{symbol}'...")
                    self.get_data(name, p)
                else:
                    self.logger.info(f"'{name}' for '{symbol}' already cached at {filepath}. Skipping download.")

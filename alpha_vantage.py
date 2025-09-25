"""
Alpha Vantage API - Main Interface

This module provides the main API function for fetching Alpha Vantage data.
It serves as a clean, simple interface to the underlying client and utilities.
"""
import pandas as pd
from typing import Optional, Dict, List
import logging

from client import AlphaVantageClient
from utils import read_stock_symbols
from alpha_vantage_schema import SYMBOL_ENDPOINTS, MACRO_ENDPOINTS

logger = logging.getLogger(__name__)


def fetch_alpha_vantage_data(symbols: Optional[List[str]] = None,
                             endpoints: Optional[Dict[str, Dict]] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             force_refresh: bool = False,
                             api_key: Optional[str] = None,
                             **client_kwargs) -> pd.DataFrame:
    """
    Fetch, process, and combine data from Alpha Vantage for specified symbols and endpoints.
    
    This function orchestrates the entire data retrieval process, from client
    initialization to fetching, caching, and combining various datasets into a
    single, unified DataFrame.
    
    Args:
        symbols: List of stock symbols. If None, reads from stocks.txt.
        endpoints: Dictionary of endpoints and their parameters.
        start_date: Start date for filtering (YYYY-MM-DD).
        end_date: End date for filtering (YYYY-MM-DD).
        force_refresh: If True, bypass cache and fetch fresh data.
        api_key: Alpha Vantage API key (overrides environment variable).
        **client_kwargs: Additional arguments for AlphaVantageClient.
        
    Returns:
        A combined DataFrame with all requested data.
    """
    try:
        client = AlphaVantageClient(api_key=api_key, **client_kwargs)
        
        if symbols is None:
            symbols = read_stock_symbols()
        
        if not endpoints:
            logger.warning("No endpoints specified, returning empty DataFrame.")
            return pd.DataFrame()
            
        all_dfs = []

        # Process symbol-specific endpoints
        symbol_endpoints = {k: v for k, v in endpoints.items() if k in SYMBOL_ENDPOINTS}
        for symbol in symbols:
            for endpoint_name, params in symbol_endpoints.items():
                params_with_symbol = {"symbol": symbol, **params}
                df = client.get_data(endpoint_name, params_with_symbol, force_refresh)
                if not df.empty:
                    df["symbol"] = symbol
                    all_dfs.append(df)

        # Process macro-economic endpoints
        macro_endpoints_map = {k: v for k, v in endpoints.items() if k in MACRO_ENDPOINTS}
        for endpoint_name, params in macro_endpoints_map.items():
            df = client.get_data(endpoint_name, params, force_refresh)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            logger.warning("No data fetched for the given symbols and endpoints.")
            return pd.DataFrame()

        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, sort=False)

        # Filter by date range if specified
        if 'date' in combined_df.index.names:
            if start_date:
                combined_df = combined_df[combined_df.index >= start_date]
            if end_date:
                combined_df = combined_df[combined_df.index <= end_date]
        
        logger.info(f"Successfully built dataset with {len(combined_df)} rows.")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error in fetch_alpha_vantage_data: {e}", exc_info=True)
        return pd.DataFrame()


def get_client(api_key: Optional[str] = None, **kwargs) -> AlphaVantageClient:
    """
    Get a configured AlphaVantageClient instance.
    
    Args:
        api_key: Alpha Vantage API key
        **kwargs: Additional arguments passed to AlphaVantageClient
        
    Returns:
        Configured AlphaVantageClient instance
    """
    return AlphaVantageClient(api_key=api_key, **kwargs)

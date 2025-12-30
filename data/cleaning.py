"""
Data Cleaning and Transformation Rules

This module contains specific cleaning rules for Alpha Vantage API endpoints.
It is used to override or refine data parsing before saving to the database.
"""
import pandas as pd
import logging
from typing import Dict, Callable

logger = logging.getLogger(__name__)

import data.utils as data_utils
import data.alpha_vantage_schema as avs

def clean_insider_transactions(df: pd.DataFrame, endpoint_name: str = "INSIDER_TRANSACTIONS") -> pd.DataFrame:
    """
    Cleans INSIDER_TRANSACTIONS data by:
    1. Renaming columns to match DB schema (handling camelCase inputs).
    2. Aggregating duplicate rows (summing shares).
    """
    if df.empty:
        return df

    if df.index.name == 'dt':
        df.reset_index(inplace=True)

    # 1. Rename columns
    rename_map = {
        "executive": "reportingPerson",
        "acquisitionOrDisposal": "transactionType",
        "sharePrice": "price"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 2. Aggregation Logic
    # PK: symbol, dt, reportingPerson, transactionType, securityType, shares, price
    group_cols = [
        col for col in df.columns 
        if col in [
            'symbol', 'dt', 'reportingPerson', 'transactionType', 'securityType', 'price'
        ]
    ]

    if 'shares' not in df.columns:
        return df
        
    initial_count = len(df)
    
    agg_dict = {'shares': 'sum'}
    
    for col in df.columns:
        if col not in group_cols and col != 'shares':
            agg_dict[col] = 'first'
            
    df_cleaned = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    final_count = len(df_cleaned)
    if final_count < initial_count:
        logger.info(f"Aggregated INSIDER_TRANSACTIONS: {initial_count} -> {final_count} rows (merged {initial_count - final_count} duplicates)")
        
    return df_cleaned

def clean_macro(df: pd.DataFrame, endpoint_name: str) -> pd.DataFrame:
    """
    Cleans MACRO data (Economic Indicators & Commodities).
    
    Logic:
    1. Rename the value column to match the endpoint name (camelCased).
    2. Ensure only [dt, value_col] are kept.
    """
    if df.empty:
        return df
        
    col_name = data_utils.normalize_to_camel_case(endpoint_name)
    
    # Identify value column (anything not dt/date)
    # The dataframe likely has camelCase columns now.
    candidates = [c for c in df.columns if c not in ["dt", "date"]]
    
    if col_name not in df.columns and candidates:
        # Rename the first non-date column to the target column name
        df.rename(columns={candidates[0]: col_name}, inplace=True)
        
    if col_name in df.columns:
        # Check if dt is in index or columns
        if "dt" not in df.columns and (df.index.name == "dt" or "dt" in df.index.names):
             df.reset_index(inplace=True)
        elif "dt" not in df.columns and df.index.name is None:
             # Just in case it's unnamed index
             df.reset_index(inplace=True)
             if "index" in df.columns:
                 df.rename(columns={"index": "dt"}, inplace=True)
                 
        # Keep only dt and the value column
        cols_to_keep = ["dt", col_name]
        
        # Verify columns exist before filtering
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        if len(existing_cols) == len(cols_to_keep):
            df = df[cols_to_keep] 
        else:
            logger.warning(f"Missing expected columns for {endpoint_name}. Expected {cols_to_keep}, found {df.columns}")
            
    return df

# Registry of cleaning functions by endpoint name
CLEANING_REGISTRY: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
    "INSIDER_TRANSACTIONS": clean_insider_transactions,
}

# Add Macro endpoints to registry
for endpoint in avs.MACRO_ENDPOINTS:
    CLEANING_REGISTRY[endpoint] = clean_macro

def clean_data(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies registered cleaning function for the given endpoint.
    """
    cleaner = CLEANING_REGISTRY.get(endpoint_name)
    if cleaner:
        try:
            return cleaner(df, endpoint_name)
        except Exception as e:
            logger.error(f"Error cleaning data for {endpoint_name}: {e}")
            return df
    return df


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.cleaning import clean_insider_transactions, clean_macro

def test_cleaning():
    print("--- Testing INSIDER_TRANSACTIONS ---")
    # Simulate duplicate data with raw API column names (camelCased as per flow)
    # Note: 'transaction_date' is usually 'dt' by the time it reaches clean_data 
    # but let's assume standard columns are there.
    data = [
        # Duplicate pair 1
        {
            'symbol': 'GOOG', 'dt': '2025-12-15', 'executive': 'ARNOLD, FRANCES', 
            'acquisitionOrDisposal': 'A', 'securityType': 'Class C Google Stock Units', 'sharePrice': 0.0,
            'shares': 50
        },
        {
            'symbol': 'GOOG', 'dt': '2025-12-15', 'executive': 'ARNOLD, FRANCES', 
            'acquisitionOrDisposal': 'A', 'securityType': 'Class C Google Stock Units', 'sharePrice': 0.0,
            'shares': 50
        }
    ]
    
    df = pd.DataFrame(data)
    print("Original DataFrame (Insider):")
    print(df)
    
    cleaned_df = clean_insider_transactions(df)
    
    print("\nCleaned DataFrame (Insider):")
    print(cleaned_df)
    
    assert len(cleaned_df) == 1, f"Expected 1 row, got {len(cleaned_df)}"
    row = cleaned_df.iloc[0]
    assert row['shares'] == 100, "Aggregation failed"
    assert 'reportingPerson' in cleaned_df.columns, "Renaming failed (executive)"
    assert 'transactionType' in cleaned_df.columns, "Renaming failed (transactionType)"
    
    print("Insider Verification Passed!")

    print("\n--- Testing MACRO (REAL_GDP) ---")
    # Simulate Macro data
    macro_data = [
        {'dt': '2025-01-01', 'value': 23000.5},
        {'dt': '2024-01-01', 'value': 22000.1}
    ]
    df_macro = pd.DataFrame(macro_data)
    print("Original DataFrame (Macro):")
    print(df_macro)
    
    cleaned_macro = clean_macro(df_macro, "REAL_GDP")
    print("\nCleaned DataFrame (Macro):")
    print(cleaned_macro)
    
    assert 'realGdp' in cleaned_macro.columns, "Renaming failed (realGdp)"
    assert 'value' not in cleaned_macro.columns, "Old column remained"
    
    print("Macro Verification Passed!")

if __name__ == "__main__":
    test_cleaning()

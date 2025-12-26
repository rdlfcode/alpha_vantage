import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import utils and settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.utils
import data.alpha_vantage_schema as avs

class TestSchemaEnforcement(unittest.TestCase):
    def setUp(self):
        # Define a mock schema (conceptually, we use an existing one from avs to strict test)
        # We'll use TIME_SERIES_DAILY as it has standard fields
        self.table_name = "TIME_SERIES_DAILY"
        
    def test_numeric_enforcement(self):
        data = {
            "symbol": ["IBM", "MSFT", "GOOG"],
            "open": ["150.5", 200.0, "None"], # Mixed string/float/None string
            "volume": ["1000", 5000, "invalid"], # Integer string, int, invalid
            "dt": ["2023-01-01", "2023-01-02", "2023-01-03"]
        }
        df = pd.DataFrame(data)
        
        # Pre-check types (they are object mostly)
        self.assertTrue(df["open"].dtype == 'object')
        
        # Apply schema
        df_clean = utils.enforce_schema(df, self.table_name)
        
        # Check 'open' is float/numeric
        self.assertTrue(pd.api.types.is_float_dtype(df_clean["open"].dtype))
        self.assertEqual(df_clean["open"].iloc[0], 150.5)
        self.assertTrue(np.isnan(df_clean["open"].iloc[2])) # "None" should be NaN
        
        # Check 'volume' is numeric (might be float if NaNs introduced, or int/float)
        # Typically pandas usage of to_numeric with NaNs results in float
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean["volume"].dtype))
        self.assertEqual(df_clean["volume"].iloc[0], 1000)
        self.assertTrue(np.isnan(df_clean["volume"].iloc[2])) # "invalid" should be NaN

    def test_date_enforcement(self):
        data = {
            "symbol": ["IBM"],
            "dt": ["2023-01-01"],
            "open": [100.0]
        }
        df = pd.DataFrame(data)
        
        # Force dt to string
        df["dt"] = df["dt"].astype(str)
        
        df_clean = utils.enforce_schema(df, self.table_name)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_clean["dt"].dtype))
        self.assertEqual(df_clean["dt"].iloc[0], pd.Timestamp("2023-01-01"))

    def test_extra_columns_ignored(self):
        # Columns not in schema should theoretically be untouched or ignored depending on implementation.
        # helper `enforce_schema` iterates over df columns. If col not in schema, it skips it (so it remains).
        data = {
            "symbol": ["IBM"],
            "random_col": ["foo"]
        }
        df = pd.DataFrame(data)
        df_clean = utils.enforce_schema(df, self.table_name)
        self.assertEqual(df_clean["random_col"].iloc[0], "foo")

if __name__ == '__main__':
    unittest.main()

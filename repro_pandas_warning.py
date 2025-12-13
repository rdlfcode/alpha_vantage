import pandas as pd
import warnings

# Filter warnings to ensure we catch them
warnings.simplefilter('always', FutureWarning)

def test_column_renaming():
    print("Testing Column Renaming...")
    # Simulate a dataframe with "1. open", "2. high" style columns
    data = {
        "1. open": [1.0, 2.0],
        "2. high": [1.1, 2.1]
    }
    df = pd.DataFrame(data)
    
    # New method using rename
    df.rename(
        columns=lambda c: c.split(". ")[1] if ". " in c else c,
        inplace=True,
    )
    print("Result columns:", df.columns.tolist())
    # assertions
    assert "open" in df.columns
    assert "high" in df.columns
    print("Column renaming test passed.")

def test_fallback_creation():
    print("\nTesting Fallback Creation...")
    # Simulate data for fallback
    data = {
        "Symbol": "IBM",
        "AssetType": "Common Stock",
        "Name": "International Business Machines",
        "Description": "None",
        "Exchange": "NYSE",
        "Currency": "USD",
        "Country": "USA",
        "Sector": "TECHNOLOGY",
        "Industry": "COMPUTER & OFFICE EQUIPMENT",
        "Address": "1 NEW ORCHARD ROAD, ARMONK, NY, US",
        "FiscalYearEnd": "December",
        "LatestQuarter": "2023-09-30",
        "MarketCapitalization": "150000000000",
        "EBITDA": "12000000000",
        "PERatio": "20.5",
        "PEGRatio": "1.2",
        "BookValue": "30.5",
        "DividendPerShare": "6.6",
        "DividendYield": "0.04",
        "EPS": "7.5",
        "RevenueTTM": "60000000000",
        "ProfitMargin": "0.1",
        "OperatingMarginTTM": "0.15",
        "ReturnOnAssetsTTM": "0.05",
        "ReturnOnEquityTTM": "0.20",
        "RevenuePerShareTTM": "65.0",
        "QuarterlyRevenueGrowthYOY": "0.02",
        "QuarterlyEarningsGrowthYOY": "0.03",
        "AnalystTargetPrice": "150.0",
        "TrailingPE": "25.0",
        "ForwardPE": "18.0",
        "PriceToSalesRatioTTM": "2.5",
        "PriceToBookRatio": "5.0",
        "EVToRevenue": "3.0",
        "EVToEBITDA": "15.0",
        "Beta": "0.9",
        "52WeekHigh": "160.0",
        "52WeekLow": "120.0",
        "50DayMovingAverage": "140.0",
        "200DayMovingAverage": "135.0",
        "SharesOutstanding": "900000000",
        "DividendDate": "2023-12-10",
        "ExDividendDate": "2023-11-10"
    }

    # New method
    df = pd.DataFrame([data])
    
    print("Result shape:", df.shape)
    print("Result columns sample:", df.columns[:5].tolist())
    print("Result index:", df.index)
    
    assert len(df) == 1
    assert "Symbol" in df.columns
    assert df.iloc[0]["Symbol"] == "IBM"
    print("Fallback creation test passed.")

def test_timeseries_creation():
    print("\nTesting TimeSeries Creation (from_dict orient='index')...")
    # Simulate time series data: {"2023-10-27": {"1. open": "100", ...}}
    # Keys are strings that look like dates? Or maybe just numbers?
    # Trying numeric-like strings just in case
    data = {
        "2023-10-27": {"1. open": "130.00", "5. volume": "1000"},
        "2023-10-26": {"1. open": "129.00", "5. volume": "1200"}
    }
    
    # Logic from alpha_vantage.py line 210
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    df.index.name = "dt"
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Also test with numeric strings as keys, which is the specific warning trigger condition
    data_numeric_keys = {
        "1": {"col": "val"},
        "2": {"col": "val"}
    }
    print("Testing numeric string keys...")
    df2 = pd.DataFrame.from_dict(data_numeric_keys, orient="index")
    
    print("TimeSeries creation test passed.")

if __name__ == "__main__":
    test_column_renaming()
    test_fallback_creation()
    test_timeseries_creation()

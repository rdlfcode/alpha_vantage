settings = {
    "data_dir": "data"
}

dataset = {
    "TIME_SERIES_DAILY": {
        "symbol": "string",
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"]
    },
    "INSIDER_TRANSACTIONS": {
         "symbol": "string", # e.g., "IBM"
    },
    # Fundamental Data ########################################################

    "OVERVIEW": {
        "symbol": "string" # e.g., "IBM"
    },
    "DIVIDENDS": {
        "symbol": "string" # e.g., "IBM"
    },
    "INCOME_STATEMENT": {
        "symbol": "string" # e.g., "IBM"
    },
    "BALANCE_SHEET": {
        "symbol": "string" # e.g., "IBM"
    },
    "CASH_FLOW": {
        "symbol": "string" # e.g., "IBM"
    },
    "EARNINGS": {
        "symbol": "string" # e.g., "IBM"
    },
    "WTI": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"]
    },
    "BRENT": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"]
    },
    "NATURAL_GAS": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"]
    },
    "COPPER": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "ALUMINUM": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "WHEAT": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "CORN": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "COTTON": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "SUGAR": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "COFFEE": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "ALL_COMMODITIES": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "REAL_GDP": {
        "interval": ["quarterly", "annual"],
        "datatype": ["json", "csv"]
    },
    "REAL_GDP_PER_CAPITA": {
        "datatype": ["json", "csv"]
    },
    "TREASURY_YIELD": {
        "interval": ["daily", "weekly", "monthly"],
        "maturity": ["3month", "2year", "5year", "7year", "10year", "30year"],
        "datatype": ["json", "csv"]
    },
    "FEDERAL_FUNDS_RATE": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"]
    },
    "CPI": {
        "interval": ["monthly", "semiannual"],
        "datatype": ["json", "csv"]
    },
    "INFLATION": {
        "datatype": ["json", "csv"]
    },
    "RETAIL_SALES": {
        "datatype": ["json", "csv"]
    },
    "DURABLES": {
        "datatype": ["json", "csv"]
    },
    "UNEMPLOYMENT": {
        "datatype": ["json", "csv"]
    },
    "NONFARM_PAYROLL": {
        "datatype": ["json", "csv"]
    }
}
"""
Alpha Vantage API Schema and Constants

This module contains all the schemas, constants, and configurations for the Alpha Vantage API.
It serves as the single source of truth for API endpoints and their parameters.
"""

BASE_URL = "https://www.alphavantage.co/query"

# Define endpoint categories
SYMBOL_ENDPOINTS = [
    "TIME_SERIES_INTRADAY",
    "TIME_SERIES_DAILY",
    "TIME_SERIES_DAILY_ADJUSTED",
    "TIME_SERIES_WEEKLY",
    "TIME_SERIES_WEEKLY_ADJUSTED",
    "TIME_SERIES_MONTHLY",
    "TIME_SERIES_MONTHLY_ADJUSTED",
    "INSIDER_TRANSACTIONS", 
    "INCOME_STATEMENT",
    "BALANCE_SHEET",
    "CASH_FLOW",
    "EARNINGS"
]

MACRO_ENDPOINTS = [
    "WTI", "BRENT", "NATURAL_GAS",  # Energy
    "COPPER", "ALUMINUM",  # Metals
    "WHEAT", "CORN", "COTTON", "SUGAR", "COFFEE",  # Agriculture
    "ALL_COMMODITIES",  # Commodity index
    "REAL_GDP", "REAL_GDP_PER_CAPITA",  # GDP
    "TREASURY_YIELD", "FEDERAL_FUNDS_RATE",  # Rates
    "CPI", "INFLATION",  # Inflation
    "RETAIL_SALES", "DURABLES",  # Consumer
    "UNEMPLOYMENT", "NONFARM_PAYROLL"  # Employment
]

ALPHA_VANTAGE_SCHEMA = {
    # Core Stock APIs
    "TIME_SERIES_INTRADAY": {
        "symbol": "string",
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "adjusted": ["true", "false"],
        "extended_hours": ["true", "false"],
        "month": "string",  # YYYY-MM format
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_DAILY": {
        "symbol": "string",
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_DAILY_ADJUSTED": {
        "symbol": "string",
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_WEEKLY": {
        "symbol": "string",
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_WEEKLY_ADJUSTED": {
        "symbol": "string",
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_MONTHLY": {
        "symbol": "string",
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_MONTHLY_ADJUSTED": {
        "symbol": "string",
        "datatype": ["json", "csv"],
    },
    "INSIDER_TRANSACTIONS": {
        "symbol": "string",
    },
    "INCOME_STATEMENT": {
        "symbol": "string",
    },
    "BALANCE_SHEET": {
        "symbol": "string",
    },
    "CASH_FLOW": {
        "symbol": "string",
    },
    "EARNINGS": {
        "symbol": "string",
    },
    # Commodities
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
        "datatype": ["json", "csv"],
    },
    "COPPER": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "ALUMINUM": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "WHEAT": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "CORN": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "COTTON": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "SUGAR": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "COFFEE": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    "ALL_COMMODITIES": {
        "interval": ["monthly", "quarterly", "annual"],
        "datatype": ["json", "csv"],
    },
    # Economic Indicators
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
        "datatype": ["json", "csv"],
    },
    "FEDERAL_FUNDS_RATE": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"],
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
    },
}
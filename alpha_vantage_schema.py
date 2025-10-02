"""
Alpha Vantage API Schema and Constants

This module contains all the schemas, constants, and configurations for the Alpha Vantage API.
It serves as the single source of truth for API endpoints and their parameters.
"""

BASE_URL = "https://www.alphavantage.co/query"

SYMBOL_ENDPOINTS = [
    # Core Time Series
    "TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED",
    "TIME_SERIES_WEEKLY", "TIME_SERIES_WEEKLY_ADJUSTED", "TIME_SERIES_MONTHLY",
    "TIME_SERIES_MONTHLY_ADJUSTED", "GLOBAL_QUOTE",
    # Fundamental Data
    "OVERVIEW", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS",
    "LISTING_STATUS", "EARNINGS_CALENDAR", "IPO_CALENDAR",
    # Alpha Intelligence
    "NEWS_SENTIMENT", "INSIDER_TRANSACTIONS",
    # Technical Indicators (a selection, as most are generic)
    "SMA", "EMA", "VWAP", "MACD", "RSI", "BBANDS"
]

MACRO_ENDPOINTS = [
    # Commodities
    "WTI", "BRENT", "NATURAL_GAS", "COPPER", "ALUMINUM", "WHEAT", "CORN",
    "COTTON", "SUGAR", "COFFEE", "ALL_COMMODITIES",
    # Economic Indicators
    "REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE",
    "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL",
    # Market Status
    "MARKET_STATUS"
]

PREMIUM_ENDPOINTS = [
    "TIME_SERIES_DAILY_ADJUSTED", "REALTIME_BULK_QUOTES", "REALTIME_OPTIONS",
    "FX_INTRADAY", "CRYPTO_INTRADAY", "ANALYTICS_FIXED_WINDOW",
    "ANALYTICS_SLIDING_WINDOW", "VWAP", "MACD"
]

DEFAULT_ENDPOINTS = {
    # == Symbol Endpoints ==
    "TIME_SERIES_DAILY": {"symbol": None, "outputsize": "compact", "datatype": "csv"},
    "INSIDER_TRANSACTIONS": {"symbol": None},
    "INCOME_STATEMENT": {"symbol": None},
    "BALANCE_SHEET": {"symbol": None},
    "CASH_FLOW": {"symbol": None},
    "EARNINGS": {"symbol": None},
    # == Macro Endpoints ==
    "WTI": {"interval": "daily", "datatype": "csv"},
    "BRENT": {"interval": "daily", "datatype": "csv"},
    "NATURAL_GAS": {"interval": "daily", "datatype": "csv"},
    "COPPER": {"interval": "monthly", "datatype": "csv"},
    "ALUMINUM": {"interval": "monthly", "datatype": "csv"},
    "WHEAT": {"interval": "monthly", "datatype": "csv"},
    "CORN": {"interval": "monthly", "datatype": "csv"},
    "COTTON": {"interval": "monthly", "datatype": "csv"},
    "SUGAR": {"interval": "monthly", "datatype": "csv"},
    "COFFEE": {"interval": "monthly", "datatype": "csv"},
    "ALL_COMMODITIES": {"interval": "monthly", "datatype": "csv"},
    "REAL_GDP": {"interval": "quarterly", "datatype": "csv"},
    "REAL_GDP_PER_CAPITA": {"datatype": "csv"},
    "TREASURY_YIELD": {"interval": "daily", "maturity": "3month", "datatype": "csv"},
    "FEDERAL_FUNDS_RATE": {"interval": "daily", "datatype": "csv"},
    "CPI": {"interval": "monthly", "datatype": "csv"},
    "INFLATION": {"datatype": "csv"},
    "RETAIL_SALES": {"datatype": "csv"},
    "DURABLES": {"datatype": "csv"},
    "UNEMPLOYMENT": {"datatype": "csv"},
    "NONFARM_PAYROLL": {"datatype": "csv"},
}

ALPHA_VANTAGE_SCHEMA = {
    # Core Stock APIs
    "TIME_SERIES_INTRADAY": {
        "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min"],
        "adjusted": ["true", "false"], "extended_hours": ["true", "false"],
        "month": "string", "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "TIME_SERIES_DAILY": {
        "symbol": "string", "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "TIME_SERIES_DAILY_ADJUSTED": {
        "symbol": "string", "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "TIME_SERIES_WEEKLY": {"symbol": "string", "datatype": ["csv", "json"]},
    "TIME_SERIES_WEEKLY_ADJUSTED": {"symbol": "string", "datatype": ["csv", "json"]},
    "TIME_SERIES_MONTHLY": {"symbol": "string", "datatype": ["csv", "json"]},
    "TIME_SERIES_MONTHLY_ADJUSTED": {"symbol": "string", "datatype": ["csv", "json"]},
    "GLOBAL_QUOTE": {"symbol": "string", "datatype": ["csv", "json"]},
    "REALTIME_BULK_QUOTES": {"symbols": "string", "datatype": ["csv", "json"]},
    "SYMBOL_SEARCH": {"keywords": "string", "datatype": ["csv", "json"]},
    "MARKET_STATUS": {},
    # Options APIs ------------------------------------------------------------
    "REALTIME_OPTIONS": {
        "symbol": "string", "require_greeks": ["false", "true"],
        "contract": "string", "datatype": ["csv", "json"],
    },
    "HISTORICAL_OPTIONS": {"symbol": "string", "date": "string", "datatype": ["csv", "json"]},
    # Alpha Intelligence ------------------------------------------------------
    "NEWS_SENTIMENT": {
        "tickers": "string", "topics": "string", "time_from": "string",
        "time_to": "string", "sort": ["LATEST", "EARLIEST", "RELEVANCE"], "limit": "integer",
    },
    "EARNINGS_CALL_TRANSCRIPT": {"symbol": "string", "quarter": "string"},
    "TOP_GAINERS_LOSERS": {},
    "INSIDER_TRANSACTIONS": {"symbol": "string"},
    "ANALYTICS_FIXED_WINDOW": {
        "SYMBOLS": "string", "RANGE": "string", "OHLC": ["open", "high", "low", "close"],
        "INTERVAL": ["1min", "5min", "15min", "30min", "60min", "DAILY", "WEEKLY", "MONTHLY"],
        "CALCULATIONS": "string",
    },
    "ANALYTICS_SLIDING_WINDOW": {
        "SYMBOLS": "string", "RANGE": "string", "OHLC": ["open", "high", "low", "close"],
        "INTERVAL": ["1min", "5min", "15min", "30min", "60min", "DAILY", "WEEKLY", "MONTHLY"],
        "WINDOW_SIZE": "integer", "CALCULATIONS": "string",
    },
    # Fundamental Data --------------------------------------------------------
    "OVERVIEW": {"symbol": "string"},
    "ETF_PROFILE": {"symbol": "string"},
    "DIVIDENDS": {"symbol": "string", "datatype": ["csv", "json"]},
    "SPLITS": {"symbol": "string", "datatype": ["csv", "json"]},
    "INCOME_STATEMENT": {"symbol": "string"},
    "BALANCE_SHEET": {"symbol": "string"},
    "CASH_FLOW": {"symbol": "string"},
    "SHARES_OUTSTANDING": {"symbol": "string", "datatype": ["csv", "json"]},
    "EARNINGS": {"symbol": "string"},
    "EARNINGS_ESTIMATES": {"symbol": "string"},
    "LISTING_STATUS": {"date": "string", "state": ["active", "delisted"]},
    "EARNINGS_CALENDAR": {"symbol": "string", "horizon": ["3month", "6month", "12month"]},
    "IPO_CALENDAR": {},
    # Forex (FX) --------------------------------------------------------------
    "CURRENCY_EXCHANGE_RATE": {"from_currency": "string", "to_currency": "string"},
    "FX_INTRADAY": {
        "from_symbol": "string", "to_symbol": "string",
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "FX_DAILY": {
        "from_symbol": "string", "to_symbol": "string",
        "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "FX_WEEKLY": {"from_symbol": "string", "to_symbol": "string", "datatype": ["csv", "json"]},
    "FX_MONTHLY": {"from_symbol": "string", "to_symbol": "string", "datatype": ["csv", "json"]},
    # Digital & Crypto Currencies ---------------------------------------------
    "CRYPTO_INTRADAY": {
        "symbol": "string", "market": "string",
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "outputsize": ["compact", "full"], "datatype": ["csv", "json"],
    },
    "DIGITAL_CURRENCY_DAILY": {"symbol": "string", "market": "string"},
    "DIGITAL_CURRENCY_WEEKLY": {"symbol": "string", "market": "string"},
    "DIGITAL_CURRENCY_MONTHLY": {"symbol": "string", "market": "string"},
    # Commodities -------------------------------------------------------------
    "WTI": {"interval": ["daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "BRENT": {"interval": ["daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "NATURAL_GAS": {"interval": ["daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "COPPER": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "ALUMINUM": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "WHEAT": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "CORN": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "COTTON": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "SUGAR": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "COFFEE": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    "ALL_COMMODITIES": {"interval": ["monthly", "quarterly", "annual"], "datatype": ["csv", "json"]},
    # Economic Indicators -----------------------------------------------------
    "REAL_GDP": {"interval": ["quarterly", "annual"], "datatype": ["csv", "json"]},
    "REAL_GDP_PER_CAPITA": {"datatype": ["csv", "json"]},
    "TREASURY_YIELD": {
        "interval": ["daily", "weekly", "monthly"],
        "maturity": ["3month", "2year", "5year", "7year", "10year", "30year"],
        "datatype": ["csv", "json"],
    },
    "FEDERAL_FUNDS_RATE": {"interval": ["daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "CPI": {"interval": ["monthly", "semiannual"], "datatype": ["csv", "json"]},
    "INFLATION": {"datatype": ["csv", "json"]},
    "RETAIL_SALES": {"datatype": ["csv", "json"]},
    "DURABLES": {"datatype": ["csv", "json"]},
    "UNEMPLOYMENT": {"datatype": ["csv", "json"]},
    "NONFARM_PAYROLL": {"datatype": ["csv", "json"]},
    # Technical Indicators ----------------------------------------------------
    "SMA": {"function": "SMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "EMA": {"function": "EMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "WMA": {"function": "WMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "DEMA": {"function": "DEMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "TEMA": {"function": "TEMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "TRIMA": {"function": "TRIMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "KAMA": {"function": "KAMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "MAMA": {"function": "MAMA", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "fastlimit": "float", "slowlimit": "float", "datatype": ["csv", "json"]},
    "VWAP": {"function": "VWAP", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min"], "datatype": ["csv", "json"]},
    "T3": {"function": "T3", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "MACD": {"function": "MACD", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "fastperiod": "integer", "slowperiod": "integer", "signalperiod": "integer", "datatype": ["csv", "json"]},
    "MACDEXT": {"function": "MACDEXT", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "fastperiod": "integer", "slowperiod": "integer", "signalperiod": "integer", "fastmatype": "integer", "slowmatype": "integer", "signalmatype": "integer", "datatype": ["csv", "json"]},
    "STOCH": {"function": "STOCH", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "fastkperiod": "integer", "slowkperiod": "integer", "slowdperiod": "integer", "slowkmatype": "integer", "slowdmatype": "integer", "datatype": ["csv", "json"]},
    "STOCHF": {"function": "STOCHF", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "fastkperiod": "integer", "fastdperiod": "integer", "fastdmatype": "integer", "datatype": ["csv", "json"]},
    "RSI": {"function": "RSI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "STOCHRSI": {"function": "STOCHRSI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "fastkperiod": "integer", "fastdperiod": "integer", "fastdmatype": "integer", "datatype": ["csv", "json"]},
    "WILLR": {"function": "WILLR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "ADX": {"function": "ADX", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "ADXR": {"function": "ADXR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "APO": {"function": "APO", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "fastperiod": "integer", "slowperiod": "integer", "matype": "integer", "datatype": ["csv", "json"]},
    "PPO": {"function": "PPO", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "fastperiod": "integer", "slowperiod": "integer", "matype": "integer", "datatype": ["csv", "json"]},
    "MOM": {"function": "MOM", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "BOP": {"function": "BOP", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "CCI": {"function": "CCI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "CMO": {"function": "CMO", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "ROC": {"function": "ROC", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "ROCR": {"function": "ROCR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "AROON": {"function": "AROON", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "AROONOSC": {"function": "AROONOSC", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "MFI": {"function": "MFI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "TRIX": {"function": "TRIX", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "ULTOSC": {"function": "ULTOSC", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "timeperiod1": "integer", "timeperiod2": "integer", "timeperiod3": "integer", "datatype": ["csv", "json"]},
    "DX": {"function": "DX", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "MINUS_DI": {"function": "MINUS_DI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "PLUS_DI": {"function": "PLUS_DI", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "MINUS_DM": {"function": "MINUS_DM", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "PLUS_DM": {"function": "PLUS_DM", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "BBANDS": {"function": "BBANDS", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "nbdevup": "integer", "nbdevdn": "integer", "matype": "integer", "datatype": ["csv", "json"]},
    "MIDPOINT": {"function": "MIDPOINT", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "MIDPRICE": {"function": "MIDPRICE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "SAR": {"function": "SAR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "acceleration": "float", "maximum": "float", "datatype": ["csv", "json"]},
    "TRANGE": {"function": "TRANGE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "ATR": {"function": "ATR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "NATR": {"function": "NATR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "time_period": "integer", "datatype": ["csv", "json"]},
    "AD": {"function": "AD", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "ADOSC": {"function": "ADOSC", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "fastperiod": "integer", "slowperiod": "integer", "datatype": ["csv", "json"]},
    "OBV": {"function": "OBV", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "datatype": ["csv", "json"]},
    "HT_TRENDLINE": {"function": "HT_TRENDLINE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "HT_SINE": {"function": "HT_SINE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "HT_TRENDMODE": {"function": "HT_TRENDMODE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "HT_DCPERIOD": {"function": "HT_DCPERIOD", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "HT_DCPHASE": {"function": "HT_DCPHASE", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
    "HT_PHASOR": {"function": "HT_PHASOR", "symbol": "string", "interval": ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"], "series_type": ["close", "open", "high", "low"], "datatype": ["csv", "json"]},
}

TABLE_SCHEMAS = {
    "CoreStockAPIs": {
        "TIME_SERIES_INTRADAY": """
CREATE TABLE IF NOT EXISTS OHLCV_INTRADAY (
    dt TIMESTAMP,
    open DECIMAL(20, 4),
    high DECIMAL(20, 4),
    low DECIMAL(20, 4),
    close DECIMAL(20, 4),
    volume INT
);""",
        "TIME_SERIES_DAILY": """
CREATE TABLE IF NOT EXISTS OHLCV_DAILY (
    dt TIMESTAMP,
    open DECIMAL(20, 4),
    high DECIMAL(20, 4),
    low DECIMAL(20, 4),
    close DECIMAL(20, 4),
    volume INT
);""",
        "GLOBAL_QUOTE": """
CREATE TABLE IF NOT EXISTS GLOBAL_QUOTE (
    symbol TEXT,
    open DECIMAL(20, 4),
    high DECIMAL(20, 4),
    low DECIMAL(20, 4),
    price DECIMAL(20, 4),
    volume INT,
    latest_trading_day TIMESTAMP,
    previous_close DECIMAL(20, 4),
    change DECIMAL(20, 4),
    change_percent TEXT
);""",
        "INCOME_STATEMENT": """
CREATE TABLE IF NOT EXISTS INCOME_STATEMENT (
    fiscalDateEnding TIMESTAMP,
    reportedCurrency TEXT,
    grossProfit INT,
    totalRevenue INT,
    operatingIncome INT,
    netIncome INT
);""",
        "BALANCE_SHEET": """
CREATE TABLE IF NOT EXISTS BALANCE_SHEET (
    fiscalDateEnding TIMESTAMP,
    reportedCurrency TEXT,
    totalAssets INT,
    totalLiabilities INT,
    totalShareholderEquity INT
);""",
        "CASH_FLOW": """
CREATE TABLE IF NOT EXISTS CASH_FLOW (
    fiscalDateEnding TIMESTAMP,
    reportedCurrency TEXT,
    operatingCashflow INT,
    cashflowFromInvestment INT,
    cashflowFromFinancing INT
);""",
        "INSIDER_TRANSACTIONS": """
CREATE TABLE IF NOT EXISTS INSIDER_TRANSACTIONS (
    symbol TEXT,
    transactionDate TIMESTAMP,
    reportingPerson TEXT,
    transactionType TEXT,
    shares INT,
    price DECIMAL(20, 4)
);""",
        "WTI": """
CREATE TABLE IF NOT EXISTS WTI (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "BRENT": """
CREATE TABLE IF NOT EXISTS BRENT (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "NATURAL_GAS": """
CREATE TABLE IF NOT EXISTS NATURAL_GAS (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "COPPER": """
CREATE TABLE IF NOT EXISTS COPPER (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "ALUMINUM": """
CREATE TABLE IF NOT EXISTS ALUMINUM (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "WHEAT": """
CREATE TABLE IF NOT EXISTS WHEAT (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "CORN": """
CREATE TABLE IF NOT EXISTS CORN (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "COTTON": """
CREATE TABLE IF NOT EXISTS COTTON (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "SUGAR": """
CREATE TABLE IF NOT EXISTS SUGAR (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "COFFEE": """
CREATE TABLE IF NOT EXISTS COFFEE (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "ALL_COMMODITIES": """
CREATE TABLE IF NOT EXISTS ALL_COMMODITIES (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "REAL_GDP": """
CREATE TABLE IF NOT EXISTS REAL_GDP (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "REAL_GDP_PER_CAPITA": """
CREATE TABLE IF NOT EXISTS REAL_GDP_PER_CAPITA (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "TREASURY_YIELD": """
CREATE TABLE IF NOT EXISTS TREASURY_YIELD (
    date TIMESTAMP,
    maturity TEXT,
    yield DECIMAL(20, 4)
);""",
        "FEDERAL_FUNDS_RATE": """
CREATE TABLE IF NOT EXISTS FEDERAL_FUNDS_RATE (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "CPI": """
CREATE TABLE IF NOT EXISTS CPI (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "INFLATION": """
CREATE TABLE IF NOT EXISTS INFLATION (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "RETAIL_SALES": """
CREATE TABLE IF NOT EXISTS RETAIL_SALES (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "DURABLES": """
CREATE TABLE IF NOT EXISTS DURABLES (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "UNEMPLOYMENT": """
CREATE TABLE IF NOT EXISTS UNEMPLOYMENT (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
        "NONFARM_PAYROLL": """
CREATE TABLE IF NOT EXISTS NONFARM_PAYROLL (
    date TIMESTAMP,
    value DECIMAL(20, 4)
);""",
    }
}
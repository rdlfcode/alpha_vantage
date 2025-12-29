"""
Alpha Vantage API Schema and Constants

This module contains all the schemas, constants, and configurations for the Alpha Vantage API.
It serves as the single source of truth for API endpoints and their parameters.
"""

DEFAULT_ENDPOINTS = {
   "TIME_SERIES_DAILY_ADJUSTED": {"symbol": None, "outputsize": "full", "datatype": "csv"},
   "INSIDER_TRANSACTIONS": {"symbol": None},
   "INCOME_STATEMENT": {"symbol": None},
   "BALANCE_SHEET": {"symbol": None},
   "CASH_FLOW": {"symbol": None},
   "EARNINGS": {"symbol": None},
   "OVERVIEW": {"symbol": None},
   "HISTORICAL_OPTIONS": {"symbol": None, "date": None, "datatype": "csv"},
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
   "TIME_SERIES_INTRADAY": """
CREATE TABLE TIME_SERIES_INTRADAY (
   dt TIMESTAMP,
   symbol TEXT,
   open DECIMAL(20, 4),
   high DECIMAL(20, 4),
   low DECIMAL(20, 4),
   close DECIMAL(20, 4),
   volume BIGINT,
   PRIMARY KEY (symbol, dt)
);""",
   "TIME_SERIES_DAILY": """
CREATE TABLE TIME_SERIES_DAILY (
   dt TIMESTAMP,
   symbol TEXT,
   open DECIMAL(20, 4),
   high DECIMAL(20, 4),
   low DECIMAL(20, 4),
   close DECIMAL(20, 4),
   volume BIGINT,
   PRIMARY KEY (symbol, dt)
);""",
   "TIME_SERIES_DAILY_ADJUSTED": """
CREATE TABLE TIME_SERIES_DAILY_ADJUSTED (
   dt TIMESTAMP,
   symbol TEXT,
   open DECIMAL(20, 4),
   high DECIMAL(20, 4),
   low DECIMAL(20, 4),
   close DECIMAL(20, 4),
   volume BIGINT,
   PRIMARY KEY (symbol, dt)
);""",
   "GLOBAL_QUOTE": """
CREATE TABLE GLOBAL_QUOTE (
   symbol TEXT,
   open DECIMAL(20, 4),
   high DECIMAL(20, 4),
   low DECIMAL(20, 4),
   price DECIMAL(20, 4),
   volume BIGINT,
   dt TIMESTAMP,
   previousClose DECIMAL(20, 4),
   change DECIMAL(20, 4),
   changePercent TEXT,
   PRIMARY KEY (symbol, dt)
);""",
   "INSIDER_TRANSACTIONS": """
CREATE TABLE INSIDER_TRANSACTIONS (
   dt TIMESTAMP,
   symbol TEXT,
   reportingPerson TEXT,
   transactionType TEXT,
   shares BIGINT,
   price DECIMAL(20, 4),
   PRIMARY KEY (symbol, dt, reportingPerson, transactionType, shares, price)
);""",
   "FUNDAMENTALS": """
CREATE TABLE FUNDAMENTALS (
   symbol TEXT,
   dt TIMESTAMP,
   periodType TEXT,   -- 'ANNUAL' or 'QUARTERLY'
   reportType TEXT,   -- 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS'
   metric TEXT,
   value DECIMAL(20, 4),
   PRIMARY KEY (symbol, periodType, reportType, metric, dt)
);""",
   "MACRO": """
CREATE TABLE MACRO (
   dt TIMESTAMP,
   wti DECIMAL(20, 4),
   brent DECIMAL(20, 4),
   naturalGas DECIMAL(20, 4),
   copper DECIMAL(20, 4),
   aluminum DECIMAL(20, 4),
   wheat DECIMAL(20, 4),
   corn DECIMAL(20, 4),
   cotton DECIMAL(20, 4),
   sugar DECIMAL(20, 4),
   coffee DECIMAL(20, 4),
   allCommodities DECIMAL(20, 4),
   realGdp DECIMAL(20, 4),
   realGdpPerCapita DECIMAL(20, 4),
   treasuryYield DECIMAL(20, 4),
   federalFundsRate DECIMAL(20, 4),
   cpi DECIMAL(20, 4),
   inflation DECIMAL(20, 4),
   retailSales DECIMAL(20, 4),
   durables DECIMAL(20, 4),
   unemployment DECIMAL(20, 4),
   nonfarmPayroll DECIMAL(20, 4),
   PRIMARY KEY (dt)
);""",
   "HISTORICAL_OPTIONS": """
CREATE TABLE HISTORICAL_OPTIONS (
    contractId VARCHAR(64),
    symbol TEXT,
    expiration DATE,
    strike DECIMAL(20, 4),
    type TEXT,
    last DECIMAL(20, 4),
    mark DECIMAL(20, 4),
    bid DECIMAL(20, 4),
    bidSize BIGINT,
    ask DECIMAL(20, 4),
    askSize BIGINT,
    volume BIGINT,
    openInterest BIGINT,
    dt TIMESTAMP,
    impliedVolatility DECIMAL(20, 5),
    delta DECIMAL(20, 5),
    gamma DECIMAL(20, 5),
    theta DECIMAL(20, 5),
    vega DECIMAL(20, 5),
    rho DECIMAL(20, 5),
    PRIMARY KEY (symbol, expiration, strike, type, dt)
);""",
    "OVERVIEW": """
CREATE TABLE OVERVIEW (
    symbol TEXT,
    assetType TEXT,
    name TEXT,
    description TEXT,
    cik TEXT,
    exchange TEXT,
    currency TEXT,
    country TEXT,
    sector TEXT,
    industry TEXT,
    address TEXT,
    fiscalYearEnd TEXT,
    latestQuarter DATE,
    marketCapitalization BIGINT,
    ebitda BIGINT,
    peRatio DECIMAL(20, 4),
    pegRatio DECIMAL(20, 4),
    bookValue DECIMAL(20, 4),
    dividendPerShare DECIMAL(20, 4),
    dividendYield DECIMAL(20, 4),
    eps DECIMAL(20, 4),
    revenuePerShareTtm DECIMAL(20, 4),
    profitMargin DECIMAL(20, 4),
    operatingMarginTtm DECIMAL(20, 4),
    returnOnAssetsTtm DECIMAL(20, 4),
    returnOnEquityTtm DECIMAL(20, 4),
    revenueTtm BIGINT,
    grossProfitTtm BIGINT,
    dilutedEpsTtm DECIMAL(20, 4),
    quarterlyEarningsGrowthYoy DECIMAL(20, 4),
    quarterlyRevenueGrowthYoy DECIMAL(20, 4),
    analystTargetPrice DECIMAL(20, 4),
    trailingPe DECIMAL(20, 4),
    forwardPe DECIMAL(20, 4),
    priceToSalesRatioTtm DECIMAL(20, 4),
    priceToBookRatio DECIMAL(20, 4),
    evToRevenue DECIMAL(20, 4),
    evToEbitda DECIMAL(20, 4),
    beta DECIMAL(20, 4),
    "52WeekHigh" DECIMAL(20, 4),
    "52WeekLow" DECIMAL(20, 4),
    "50DayMovingAverage" DECIMAL(20, 4),
    "200DayMovingAverage" DECIMAL(20, 4),
    sharesOutstanding BIGINT,
    dividendDate DATE,
    exDividendDate DATE,
    dt TIMESTAMP,
    PRIMARY KEY (symbol)
);""",
    "MODEL_METADATA": """
CREATE TABLE MODEL_METADATA (
    dt TIMESTAMP,
    model_name TEXT,
    model_type TEXT,
    training_run_id TEXT,
    metrics JSON,
    best_val_loss DECIMAL(20, 6),
    dataset_size INT,
    checkpoint_path TEXT,
    PRIMARY KEY (training_run_id)
);"""
}

TABLE_PKS = {
   "TIME_SERIES_INTRADAY": ["symbol", "dt"],
   "TIME_SERIES_DAILY": ["symbol", "dt"],
   "TIME_SERIES_DAILY_ADJUSTED": ["symbol", "dt"],
   "GLOBAL_QUOTE": ["symbol", "dt"],
   "INSIDER_TRANSACTIONS": ["symbol", "dt", "reportingPerson", "transactionType", "shares", "price"],
   "FUNDAMENTALS": ["symbol", "periodType", "reportType", "metric", "dt"],
   "MACRO": ["dt"],
   "HISTORICAL_OPTIONS": ["symbol", "expiration", "strike", "type", "dt"],
   "OVERVIEW": ["symbol", "dt"],
   "MODEL_METADATA": ["training_run_id"]
}

BASE_URL = "https://www.alphavantage.co/query"

FUNDAMENTAL_ENDPOINTS = ["INCOME_STATEMENT","BALANCE_SHEET","CASH_FLOW","EARNINGS"]

PREMIUM_ENDPOINTS = [
   "TIME_SERIES_DAILY_ADJUSTED", "REALTIME_BULK_QUOTES", "REALTIME_OPTIONS",
   "FX_INTRADAY", "CRYPTO_INTRADAY", "ANALYTICS_FIXED_WINDOW",
   "ANALYTICS_SLIDING_WINDOW", "VWAP", "MACD"
]

SYMBOL_ENDPOINTS = [k for k, v in ALPHA_VANTAGE_SCHEMA.items() if "symbol" in v or "symbols" in v or "Symbol" in v]

MACRO_ENDPOINTS = list(set(ALPHA_VANTAGE_SCHEMA) - set(SYMBOL_ENDPOINTS))

# Map endpoints to their tables
ENDPOINT_TO_TABLE_MAP = {}

# 1. Macro Endpoints -> MACRO table
for endpoint in MACRO_ENDPOINTS:
    ENDPOINT_TO_TABLE_MAP[endpoint] = "MACRO"

# 2. Symbol Endpoints -> Their own table (defaulting to endpoint name)
for endpoint in SYMBOL_ENDPOINTS:
    if endpoint in FUNDAMENTAL_ENDPOINTS:
        ENDPOINT_TO_TABLE_MAP[endpoint] = "FUNDAMENTALS"
    else:
        ENDPOINT_TO_TABLE_MAP[endpoint] = endpoint
        
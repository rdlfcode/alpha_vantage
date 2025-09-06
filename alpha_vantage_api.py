BASE_URL = "https://www.alphavantage.co/query"

alpha_vantage_schema = {
    # Core Stock APIs #########################################################
    "TIME_SERIES_INTRADAY": {
        "symbol": "string",  # e.g., "IBM"
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "adjusted": ["true", "false"],  # boolean represented as string
        "extended_hours": ["true", "false"],  # boolean represented as string
        "month": "string (YYYY-MM format)",  # e.g., "2009-01"
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_DAILY": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_DAILY_ADJUSTED": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_WEEKLY": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_WEEKLY_ADJUSTED": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_MONTHLY": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "datatype": ["json", "csv"],
    },
    "TIME_SERIES_MONTHLY_ADJUSTED": {
        "symbol": "string",  # e.g., "IBM", "TSCO.LON"
        "datatype": ["json", "csv"],
    },
    "GLOBAL_QUOTE": {"symbol": "string", "datatype": ["json", "csv"]},  # e.g., "IBM"
    "REALTIME_BULK_QUOTES": {  # Premium
        "symbol": "string (comma-separated, up to 100)",  # e.g., "MSFT,AAPL,IBM"
        "datatype": ["json", "csv"],
    },
    "SYMBOL_SEARCH": {
        "keywords": "string",  # e.g., "microsoft"
        "datatype": ["json", "csv"],
    },
    "MARKET_STATUS": {
        # No parameters other than function and apikey
    },
    # Options Data APIs #######################################################
    "REALTIME_OPTIONS": {  # Premium
        "symbol": "string",  # e.g., "IBM"
        "require_greeks": ["true", "false"],  # boolean represented as string
        "contract": "string (US options contract ID)",  # e.g., "IBM270115C00390000"
        "datatype": ["json", "csv"],
    },
    "HISTORICAL_OPTIONS": {
        "symbol": "string",  # e.g., "IBM"
        "date": "string (YYYY-MM-DD format, > 2008-01-01)",  # e.g., "2017-11-15"
        "datatype": ["json", "csv"],
    },
    # Alpha Intelligence ######################################################
    "NEWS_SENTIMENT": {
        "tickers": "string (comma-separated stock/crypto/forex symbols)",  # e.g., "IBM" or "COIN,CRYPTO:BTC,FOREX:USD"
        "topics": [
            "blockchain",
            "earnings",
            "ipo",
            "mergers_and_acquisitions",
            "financial_markets",
            "economy_fiscal",
            "economy_monetary",
            "economy_macro",
            "energy_transportation",
            "finance",
            "life_sciences",
            "manufacturing",
            "real_estate",
            "retail_wholesale",
            "technology",
            "string (comma-separated topics)",
        ],  # Also allows comma-separated combinations
        "time_from": "string (YYYYMMDDTHHMM format)",  # e.g., "20220410T0130"
        "time_to": "string (YYYYMMDDTHHMM format)",  # e.g., "20220411T0130"
        "sort": ["LATEST", "EARLIEST", "RELEVANCE"],
        "limit": "integer (up to 1000, default 50)",  # e.g., 50, 1000
    },
    "EARNINGS_CALL_TRANSCRIPT": {
        "symbol": "string",  # e.g., "IBM"
        "quarter": "string (YYYYQN format, >= 2010Q1)",  # e.g., "2024Q1"
    },
    "TOP_GAINERS_LOSERS": {
        # No parameters other than function and apikey
    },
    "INSIDER_TRANSACTIONS": {
        "symbol": "string",  # e.g., "IBM"
    },
    "ANALYTICS_FIXED_WINDOW": {
        "SYMBOLS": "string (comma-separated, up to 5 free/50 premium)",  # e.g., "AAPL,MSFT,IBM"
        "RANGE": [
            "full",
            "{N}day",
            "{N}week",
            "{N}month",
            "{N}year",
            "{N}minute",
            "{N}hour",  # Intraday only
            "YYYY-MM-DD",
            "YYYY-MM",  # Specific dates/months
            "YYYY-MM-DDTHH:MM:SS",
        ],  # Specific datetime, can specify start & end with two RANGE params
        "OHLC": ["open", "high", "low", "close"],
        "INTERVAL": [
            "1min",
            "5min",
            "15min",
            "30min",
            "60min",
            "DAILY",
            "WEEKLY",
            "MONTHLY",
        ],
        "CALCULATIONS": [
            "MIN",
            "MAX",
            "MEAN",
            "MEDIAN",
            "CUMULATIVE_RETURN",
            "VARIANCE",
            "VARIANCE(annualized=True)",
            "STDDEV",
            "STDDEV(annualized=True)",
            "MAX_DRAWDOWN",
            "HISTOGRAM",
            "HISTOGRAM(bins=N)",
            "AUTOCORRELATION",
            "AUTOCORRELATION(lag=N)",
            "COVARIANCE",
            "COVARIANCE(annualized=True)",
            "CORRELATION",
            "CORRELATION(method=KENDALL)",
            "CORRELATION(method=SPEARMAN)",
            "string (comma-separated calculations)",
        ],
    },
    "ANALYTICS_SLIDING_WINDOW": {
        "SYMBOLS": "string (comma-separated, up to 5 free/50 premium)",  # e.g., "AAPL,IBM"
        "RANGE": [
            "full",
            "{N}day",
            "{N}week",
            "{N}month",
            "{N}year",
            "{N}minute",
            "{N}hour",  # Intraday only
            "YYYY-MM-DD",
            "YYYY-MM",  # Specific dates/months
            "YYYY-MM-DDTHH:MM:SS",
        ],  # Specific datetime, can specify start & end with two RANGE params
        "OHLC": ["open", "high", "low", "close"],
        "INTERVAL": [
            "1min",
            "5min",
            "15min",
            "30min",
            "60min",
            "DAILY",
            "WEEKLY",
            "MONTHLY",
        ],
        "WINDOW_SIZE": "integer (>=10)",
        "CALCULATIONS": [
            "MEAN",
            "MEDIAN",
            "CUMULATIVE_RETURN",
            "VARIANCE",
            "VARIANCE(annualized=True)",
            "STDDEV",
            "STDDEV(annualized=True)",
            "COVARIANCE",
            "COVARIANCE(annualized=True)",
            "CORRELATION",
            "CORRELATION(method=KENDALL)",
            "CORRELATION(method=SPEARMAN)",
            "string (comma-separated calculations, 1 for free, multiple for premium)",
        ],
    },
    # Fundamental Data ########################################################
    "OVERVIEW": {"symbol": "string"},  # e.g., "IBM"
    "ETF_PROFILE": {"symbol": "string"},  # e.g., "QQQ"
    "DIVIDENDS": {"symbol": "string"},  # e.g., "IBM"
    "SPLITS": {"symbol": "string"},  # e.g., "IBM"
    "INCOME_STATEMENT": {"symbol": "string"},  # e.g., "IBM"
    "BALANCE_SHEET": {"symbol": "string"},  # e.g., "IBM"
    "CASH_FLOW": {"symbol": "string"},  # e.g., "IBM"
    "EARNINGS": {"symbol": "string"},  # e.g., "IBM"
    "LISTING_STATUS": {  # Returns CSV only
        "date": "string (YYYY-MM-DD format, > 2010-01-01)",  # e.g., "2013-08-03"
        "state": ["active", "delisted"],
    },
    "EARNINGS_CALENDAR": {  # Returns CSV only
        "symbol": "string",  # e.g., "IBM"
        "horizon": ["3month", "6month", "12month"],
    },
    "IPO_CALENDAR": {  # Returns CSV only
        # No parameters other than function and apikey
    },
    # Forex ###################################################################
    "CURRENCY_EXCHANGE_RATE": {
        "from_currency": "string (physical or digital/crypto symbol)",  # e.g., "USD", "BTC"
        "to_currency": "string (physical or digital/crypto symbol)",  # e.g., "JPY", "EUR"
    },
    "FX_INTRADAY": {  # Premium
        "from_symbol": "string (forex symbol)",  # e.g., "EUR"
        "to_symbol": "string (forex symbol)",  # e.g., "USD"
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "FX_DAILY": {
        "from_symbol": "string (forex symbol)",  # e.g., "EUR"
        "to_symbol": "string (forex symbol)",  # e.g., "USD"
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "FX_WEEKLY": {
        "from_symbol": "string (forex symbol)",  # e.g., "EUR"
        "to_symbol": "string (forex symbol)",  # e.g., "USD"
        "datatype": ["json", "csv"],
    },
    "FX_MONTHLY": {
        "from_symbol": "string (forex symbol)",  # e.g., "EUR"
        "to_symbol": "string (forex symbol)",  # e.g., "USD"
        "datatype": ["json", "csv"],
    },
    # Cryptocurrencies ########################################################
    "CRYPTO_INTRADAY": {  # Premium
        "symbol": "string (digital/crypto symbol)",  # e.g., "ETH"
        "market": "string (exchange market symbol)",  # e.g., "USD"
        "interval": ["1min", "5min", "15min", "30min", "60min"],
        "outputsize": ["compact", "full"],
        "datatype": ["json", "csv"],
    },
    "DIGITAL_CURRENCY_DAILY": {
        "symbol": "string (digital/crypto symbol)",  # e.g., "BTC"
        "market": "string (exchange market symbol)",  # e.g., "EUR"
        "datatype": [
            "json",
            "csv",
        ],  # Note: Doc implies only JSON/CSV but examples show datatype=csv usage
    },
    "DIGITAL_CURRENCY_WEEKLY": {
        "symbol": "string (digital/crypto symbol)",  # e.g., "BTC"
        "market": "string (exchange market symbol)",  # e.g., "EUR"
        "datatype": [
            "json",
            "csv",
        ],  # Note: Doc implies only JSON/CSV but examples show datatype=csv usage
    },
    "DIGITAL_CURRENCY_MONTHLY": {
        "symbol": "string (digital/crypto symbol)",  # e.g., "BTC"
        "market": "string (exchange market symbol)",  # e.g., "EUR"
        "datatype": [
            "json",
            "csv",
        ],  # Note: Doc implies only JSON/CSV but examples show datatype=csv usage
    },
    # Commodities #############################################################
    "WTI": {"interval": ["daily", "weekly", "monthly"], "datatype": ["json", "csv"]},
    "BRENT": {"interval": ["daily", "weekly", "monthly"], "datatype": ["json", "csv"]},
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
    # Economic Indicators #####################################################
    "REAL_GDP": {"interval": ["quarterly", "annual"], "datatype": ["json", "csv"]},
    "REAL_GDP_PER_CAPITA": {"datatype": ["json", "csv"]},
    "TREASURY_YIELD": {
        "interval": ["daily", "weekly", "monthly"],
        "maturity": ["3month", "2year", "5year", "7year", "10year", "30year"],
        "datatype": ["json", "csv"],
    },
    "FEDERAL_FUNDS_RATE": {
        "interval": ["daily", "weekly", "monthly"],
        "datatype": ["json", "csv"],
    },
    "CPI": {"interval": ["monthly", "semiannual"], "datatype": ["json", "csv"]},
    "INFLATION": {"datatype": ["json", "csv"]},
    "RETAIL_SALES": {"datatype": ["json", "csv"]},
    "DURABLES": {"datatype": ["json", "csv"]},
    "UNEMPLOYMENT": {"datatype": ["json", "csv"]},
    "NONFARM_PAYROLL": {"datatype": ["json", "csv"]},
}

premium_endpoints = [
    "TIME_SERIES_DAILY_ADJUSTED",
    "REALTIME_BULK_QUOTES",
    "REALTIME_OPTIONS",
    "FX_INTRADAY",
    "CRYPTO_INTRADAY",
    "ANALYTICS_FIXED_WINDOW",
    "ANALYTICS_SLIDING_WINDOW",
]

settings = {
   ######### DATA ##################
   "data_dir": "data",
   "data_ext": ".parquet",
   "db_path": "data/alpha_vantage.db",
   ######### ALPHA VANTAGE #########
   "AlphaVantageRPM": 75,
   "AlphaVantageRPD": 1000000,
   "MaxConcurrentRequests": 1,
   "exchange_timezones": {
        "US/Eastern": ["NYSE", "NASDAQ", "NASDAQ NMSC", "AMEX", "BATS"],
        "Europe/London": ["LSE"],
   },
   ######### LOGGING ###############
   "logging": {
      "filename": "av.log",
      "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
   },
}

settings.update({"AlphaVantagePremium": settings.get("AlphaVantageRPD", 25) > 25})
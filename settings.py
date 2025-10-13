settings = {
    ######### DATA ##################
    "data_dir": "data",
    "data_ext": ".parquet",
    "db_path": "data/alpha_vantage.db",
    ######### ALPHA VANTAGE #########
    "AlphaVantageRPM": 75,
    "AlphaVantageRPD": 25,
    "AlphaVantagePremium": False,
    ######### LOGGING ###############
    "logging": {
        "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}

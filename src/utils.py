import logging
import tomllib
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent/ "config.toml"
_loggers = {}

def parse_config(config_path = CONFIG_PATH):
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    project_root = Path(__file__).resolve().parent.parent
    for key, path in config['path'].items():
        config['path'][key] = project_root / path

    # Generate timestamped run directory inside logs/
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M")
    log_root = config['path']['log_dir']
    run_log_dir = log_root / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    config['path']['run_log_dir'] = run_log_dir

    return config


def get_logger(name: str, config):
    """
    Returns a logger that logs to a timestamped file and to the console. The name will be the name of the actual log file.
    """
    if name in _loggers:
        return _loggers[name]

    log_path = config['path']['run_log_dir'] / f"{name}.log"

    logger = logging.getLogger(name)
    # Ignore any messages below level INFO. Log INFO, WARNING, ERROR, and CRITICAL
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _loggers[name] = logger
    return logger

def load_market_data(config) -> pd.DataFrame:
    """
    Load market data from a Parquet file as a multi-indexed DataFrame. The config should contain the address of the datafile
    Expected index: (date, symbol)
    """
    return pd.read_parquet(config['path']['data_file'])
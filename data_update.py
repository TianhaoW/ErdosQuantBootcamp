import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.utils import parse_config, get_logger

# Load config
config = parse_config()
logger = get_logger("data_update", config)
symbols = config['data']['tickers']
default_start_date = config['data']['default_start_date']
data_path = config['path']['data_file'] 

# Load existing data if it exists
if os.path.exists(data_path):
    logger.info(f"Loading existing data from {data_path}")
    existing_df = pd.read_parquet(data_path)
else:
    logger.info(f"No existing data found. Creating new dataset.")
    existing_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    existing_df.index = pd.MultiIndex.from_tuples([], names=["date", "symbol"])

# Get last available date per symbol
last_date = (
    existing_df.reset_index()
    .groupby("symbol")["date"]
    .max()
    .to_dict()
)

# Target end date
today = datetime.today().date()

new_data = []

for symbol in symbols:
    start = last_date.get(symbol, datetime.strptime(default_start_date, "%Y-%m-%d").date() - timedelta(days=1)) + timedelta(days=1)

    if start > today:
        logger.info(f"{symbol} is already up to date.")
        continue

    logger.info(f"Downloading {symbol} from {start} to {today} ...")
    df = yf.download(symbol, start=start, end=today)
    if df.empty:
        logger.info(f"Warning: no data returned for {symbol}")
        continue

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df["symbol"] = symbol
    df["date"] = df.index.date
    df = df.set_index(["date", "symbol"])

    new_data.append(df)

# Merge with existing data
if new_data:
    combined = pd.concat([existing_df] + new_data)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    combined.to_parquet(data_path)
    logger.info(f"Saved updated data to {data_path}")
else:
    logger.info("No new data downloaded.")

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set the current working directory to the 'scripts' directory
script_cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_cwd)
print(f"Current working directory set to: {script_cwd}")

# Determine the base directory of the project
base_dir = os.path.abspath(os.path.join(script_cwd, '..', '..'))
print(f"Base directory determined as: {base_dir}")

# Define paths and parameters
data_path = os.path.join(base_dir, 'data', 'coinbase', 'candles', 'BTC-USD', 'FIVE_MINUTE')
output_base_path = os.path.join(script_cwd, 'test_trade_output_files')
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 6, 5)

# Ensure output directory exists
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(output_base_path, timestamp)
os.makedirs(output_path, exist_ok=True)

# Function to load data files for the given date range
def load_data_files(data_path, start_date, end_date):
    all_files = sorted(os.listdir(data_path))
    data_frames = []
    found_start_date = False
    found_end_date = False

    for file in all_files:
        try:
            parts = file.split('_')
            start_timestamp, end_timestamp = int(parts[3]), int(parts[4].split('.')[0])
        except (ValueError, IndexError) as e:
            print(f"Skipping file {file}: {e}")
            continue

        file_start_date = datetime.utcfromtimestamp(start_timestamp)
        file_end_date = datetime.utcfromtimestamp(end_timestamp)

        if file_start_date > end_date:
            break

        df = pd.read_csv(os.path.join(data_path, file))
        df.columns = ['Start', 'Low', 'High', 'Open', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Start'], unit='s')
        df = df.set_index('Date')

        if not found_start_date:
            if file_end_date >= start_date:
                df = df.loc[start_date:]
                found_start_date = True
            else:
                continue

        if found_end_date or file_end_date > end_date:
            df = df.loc[:end_date + timedelta(days=1) - timedelta(seconds=1)]
            found_end_date = True

        data_frames.append(df)

        if found_end_date:
            break

    if data_frames:
        combined_data = pd.concat(data_frames)
        # Drop rows with dates lower than start_date and higher than end_date
        combined_data = combined_data[(combined_data.index >= start_date) & (combined_data.index <= end_date)]
        # Sort the data by date
        combined_data = combined_data.sort_index()
        return combined_data
    else:
        return pd.DataFrame(columns=['Start', 'Low', 'High', 'Open', 'Close', 'Volume'])

# Load the data
data = load_data_files(data_path, start_date, end_date)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# Calculate RSI
calculate_rsi(data)

# Save RSI values to file
rsi_file = os.path.join(output_path, f'RSI_values_{timestamp}.csv')
data.to_csv(rsi_file)
print(f'RSI values saved to {rsi_file}')

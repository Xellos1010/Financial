# Commented out Boto3 initialization for future use
# import boto3
# Initialize S3 client
# s3 = boto3.client('s3')

# Commented out S3 data loading function for future use
# def load_s3_data(granularity):
#     obj = s3.get_object(Bucket=bucket_name, Key=f'{data_path}/{granularity}/data.csv')
#     df = pd.read_csv(obj['Body'])
#     return df

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
from itertools import product

# Set the current working directory to the 'scripts' directory
script_cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_cwd)
print(f"Current working directory set to: {script_cwd}")

# Determine the base directory of the project
base_dir = os.path.abspath(os.path.join(script_cwd, '..', '..', '..'))
print(f"Base directory determined as: {base_dir}")

# Define local data path using the base directory
base_data_path = os.path.join(base_dir, 'data', 'coinbase', 'candles', 'BTC-USD')
print(f"Base data path set to: {base_data_path}")

# List of granularities to train on
granularities = [
    'ONE_MINUTE',
    'FIVE_MINUTE',
    'FIFTEEN_MINUTE',
    'THIRTY_MINUTE',
    'ONE_HOUR',
    'TWO_HOUR',
    'SIX_HOUR',
    'ONE_DAY'
]

# Function to load local data
def load_local_data(granularity):
    directory = os.path.join(base_data_path, granularity)
    print(f"Loading data from directory: {directory}")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filepath}")
            df = pd.read_csv(filepath)
            dataframes.append(df)
    return pd.concat(dataframes)

# Function to calculate RSI
def calculate_rsi(df, window):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to preprocess data
def preprocess_data(df):
    print("Calculating RSI")
    df['RSI'] = calculate_rsi(df, 14)
    df.dropna(inplace=True)
    
    # Normalize/scale features
    print("Scaling features")
    scaler = StandardScaler()
    df[['Close', 'RSI']] = scaler.fit_transform(df[['Close', 'RSI']])
    
    return df, scaler

# Function to simulate trading strategy
def simulate_trading(df, rsi_window, rsi_overbought, rsi_oversold, stop_loss, take_profit):
    df['RSI'] = calculate_rsi(df, rsi_window)
    df['signal'] = 0
    df.loc[df['RSI'] < rsi_oversold, 'signal'] = 1  # Buy signal
    df.loc[df['RSI'] > rsi_overbought, 'signal'] = -1  # Sell signal

    df['position'] = df['signal'].shift(1)
    df['position'].fillna(0, inplace=True)
    
    df['entry_price'] = df['Close'].where(df['position'] != 0)
    df['entry_price'].fillna(method='ffill', inplace=True)
    df['entry_price'] = df['entry_price'].where(df['position'] != 0)
    
    df['stop_loss_price'] = df['entry_price'] * (1 - stop_loss)
    df['take_profit_price'] = df['entry_price'] * (1 + take_profit)
    
    df['exit_price'] = df['Close'].where(
        (df['Close'] <= df['stop_loss_price']) | (df['Close'] >= df['take_profit_price'])
    )
    df['exit_price'].fillna(method='ffill', inplace=True)
    df['exit_price'] = df['exit_price'].where(df['position'] != 0)
    
    df['profit'] = df['exit_price'] - df['entry_price']
    df['profit'] = df['profit'].where(df['position'] != 0)
    
    total_profit = df['profit'].sum()
    return total_profit, df

# Parameter ranges
rsi_window_range = list(range(7, 16))
rsi_overbought_range = list(range(70, 88))
rsi_oversold_range = list(range(25, 36))
stop_loss_range = [round(x, 2) for x in np.arange(0.02, 0.07, 0.01)]
take_profit_range = [round(x, 2) for x in np.arange(0.02, 0.11, 0.01)]

# Create a list to store results
results = []

# Load data for each granularity and optimize RSI parameters
for granularity in granularities:
    print(f"Processing granularity: {granularity}")
     # Check if the signal file already exists
     # Create directory if it doesn't exist
    output_dir = os.path.join(script_cwd, granularity)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the signal file already exists
    signal_file_pattern = os.path.join(output_dir, f'signals_{granularity}_*.csv')
    existing_signal_files = [f for f in os.listdir(output_dir) if f.startswith(f'signals_{granularity}_') and f.endswith('.csv')]
    if existing_signal_files:
        print(f'Signals file for {granularity} already exists. Skipping...')
        continue
    df = load_local_data(granularity)
    df, scaler = preprocess_data(df)  # Preprocess data
    
    best_profit = -np.inf
    best_params = None
    best_df = None
    
    for params in product(rsi_window_range, rsi_overbought_range, rsi_oversold_range, stop_loss_range, take_profit_range):
        rsi_window, rsi_overbought, rsi_oversold, stop_loss, take_profit = params
        profit, simulated_df = simulate_trading(df, rsi_window, rsi_overbought, rsi_oversold, stop_loss, take_profit)
        
        if profit > best_profit:
            best_profit = profit
            best_params = params
            best_df = simulated_df.copy()
    
    results.append({
        'granularity': granularity,
        'best_params': best_params,
        'best_profit': best_profit
    })
    print(f'Best parameters for {granularity}: {best_params} with profit: {best_profit}')
    
    # Save the unscaled RSI values for inspection
    df['RSI_unscaled'] = scaler.inverse_transform(df[['Close', 'RSI']])[:, 1]
    
    # Save the best signals to a file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    signal_file = os.path.join(output_dir, f'signals_{granularity}_{timestamp}.csv')
    best_df.to_csv(signal_file, index=False)
    print(f'Signals saved to {signal_file}')
    
    # Save the unscaled RSI values to a file for inspection
    rsi_file = os.path.join(output_dir, f'rsi_unscaled_{granularity}_{timestamp}.csv')
    df[['Start', 'Close', 'RSI', 'RSI_unscaled']].to_csv(rsi_file, index=False)
    print(f'RSI values saved to {rsi_file}')

# Save results to a dated file
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = os.path.join(script_cwd, f'optimization_results_{timestamp}.csv')
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f'Results saved to {output_file}')

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
output_path = script_cwd
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 6, 5)
rsi_overbought = 79
rsi_oversold = 26
rsi_window = 12
profit_exit = 0.05
loss_stop = 0.02

# Ensure output directory exists
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
calculate_rsi(data, rsi_window)

# Debug: Check RSI values
print("RSI Calculation:")
print(data[['Close', 'RSI']].head(20))

# Function to simulate the trading strategy
def simulate_strategy(data, rsi_overbought, rsi_oversold, profit_exit, loss_stop):
    data['Signal'] = 0
    data.loc[data['RSI'] < rsi_oversold, 'Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > rsi_overbought, 'Signal'] = -1  # Sell signal

    # Debug: Check signal values
    print("Signal Generation:")
    print(data[['RSI', 'Signal']].head(20))

    data['Position'] = data['Signal'].shift()
    data['Entry_Price'] = data['Close'].where(data['Position'].notnull(), np.nan)
    data['Entry_Price'] = data['Entry_Price'].ffill()

    data['Profit_Exit_Price'] = data['Entry_Price'] * (1 + profit_exit)
    data['Stop_Loss_Price'] = data['Entry_Price'] * (1 - loss_stop)

    data['Exit_Price'] = np.nan

    for i in range(1, len(data)):
        if not np.isnan(data['Position'].iloc[i]):
            entry_price = data['Entry_Price'].iloc[i]
            for j in range(i + 1, len(data)):
                close_price = data['Close'].iloc[j]
                if close_price >= entry_price * (1 + profit_exit):
                    data.at[data.index[j], 'Exit_Price'] = entry_price * (1 + profit_exit)
                    break
                elif close_price <= entry_price * (1 - loss_stop):
                    data.at[data.index[j], 'Exit_Price'] = entry_price * (1 - loss_stop)
                    break

    data['Trade_Profit'] = np.where(data['Position'] == 1, data['Exit_Price'] - data['Entry_Price'], 0)
    data['Trade_Profit'] = np.where(data['Position'] == -1, data['Entry_Price'] - data['Exit_Price'], data['Trade_Profit'])

    # Debug: Check trade profits
    print("Trade Profits:")
    print(data[['Position', 'Entry_Price', 'Profit_Exit_Price', 'Stop_Loss_Price', 'Exit_Price', 'Trade_Profit']].head(20))

    return data

# Simulate the trading strategy
strategy_results = simulate_strategy(data, rsi_overbought, rsi_oversold, profit_exit, loss_stop)

# Calculate performance metrics
total_trades = strategy_results['Signal'].abs().sum()
profitable_trades = (strategy_results['Trade_Profit'] > 0).sum()
total_profit = strategy_results['Trade_Profit'].sum()
percent_profitable = profitable_trades / total_trades * 100 if total_trades > 0 else 0
profit_factor = strategy_results['Trade_Profit'][strategy_results['Trade_Profit'] > 0].sum() / \
                abs(strategy_results['Trade_Profit'][strategy_results['Trade_Profit'] < 0].sum()) if total_trades > 0 else 0
max_drawdown = strategy_results['Trade_Profit'].cumsum().min()

# Create performance summary
performance_summary = {
    'Total Trades': total_trades,
    'Percent Profitable': percent_profitable,
    'Total Profit (USD)': total_profit,
    'Profit Factor': profit_factor,
    'Max Drawdown (USD)': max_drawdown
}

# Save strategy results and performance summary to files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
signals_file = os.path.join(output_path, f'signals_{timestamp}.csv')
performance_file = os.path.join(output_path, f'performance_summary_{timestamp}.csv')

strategy_results.to_csv(signals_file)
pd.DataFrame([performance_summary]).to_csv(performance_file, index=False)

# Print performance summary
print("Performance Summary:")
for key, value in performance_summary.items():
    print(f"{key}: {value}")

print(f"Signals saved to {signals_file}")
print(f"Performance summary saved to {performance_file}")

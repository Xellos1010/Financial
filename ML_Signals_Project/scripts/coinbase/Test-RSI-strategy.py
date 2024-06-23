import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Step 0: Initialize Script

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

# Part 1: Generate RSI Values and Save to File

start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 6, 5)
rsi_window = 12  # Input for RSI window

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
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# Calculate RSI with the given window
calculate_rsi(data, rsi_window)

# Save RSI values and window to file
rsi_file = os.path.join(output_path, f'RSI_values_{rsi_window}_{timestamp}.csv')
data.to_csv(rsi_file)
print(f'RSI values saved to {rsi_file}')

# Part 2: Generate Signals and Save to File

# Define parameters
rsi_overbought = 79
rsi_oversold = 26

# Load RSI values
rsi_file = os.path.join(output_path, f'RSI_values_{rsi_window}_{timestamp}.csv')
data = pd.read_csv(rsi_file, index_col='Date', parse_dates=True)

# Function to generate signals
def generate_signals(data, rsi_overbought, rsi_oversold):
    data['Signal'] = 0
    data.loc[data['RSI'] < rsi_oversold, 'Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > rsi_overbought, 'Signal'] = -1  # Sell signal

    # Debug: Check signal values
    print("Signal Generation:")
    print(data[['RSI', 'Signal']].head(20))

    return data

# Generate signals
data = generate_signals(data, rsi_overbought, rsi_oversold)

# Save signals to file with parameters in the filename
signals_file = os.path.join(output_path, f'signals_RSI_{rsi_window}_{rsi_overbought}_{rsi_oversold}_{timestamp}.csv')
data.to_csv(signals_file)
print(f'Signals saved to {signals_file}')

# Part 3: Execute Trades and Generate Performance File

# Define parameters
profit_exit = 0.05
loss_stop = 0.02
starting_balance = 1000  # Initial capital
max_pyramids = 1  # Maximum number of pyramiding trades
equity_percent = 1.0  # Percent of equity to use per trade (e.g., 0.5 for 50%)

# Load signals
signals_file = os.path.join(output_path, f'signals_RSI_{rsi_window}_{rsi_overbought}_{rsi_oversold}_{timestamp}.csv')
data = pd.read_csv(signals_file, index_col='Date', parse_dates=True)

# Function to execute trades with pyramiding
def execute_trades_with_pyramiding(data, profit_exit, loss_stop, starting_balance, max_pyramids, equity_percent):
    balance = starting_balance
    total_loss = 0
    active_trades = []
    trade_data = []

    for i in range(1, len(data)):
        signal = data['Signal'].iloc[i]

        if signal == 1 and len(active_trades) < max_pyramids:
            # Enter a new trade
            entry_price = data['Close'].iloc[i]
            trade_size = balance * equity_percent
            active_trades.append({
                'entry_date': data.index[i],
                'entry_price': entry_price,
                'trade_size': trade_size,
                'profit_exit_price': entry_price * (1 + profit_exit),
                'stop_loss_price': entry_price * (1 - loss_stop)
            })
            balance -= trade_size

        new_active_trades = []
        for trade in active_trades:
            close_price = data['Close'].iloc[i]
            if close_price >= trade['profit_exit_price'] or close_price <= trade['stop_loss_price']:
                # Exit the trade
                exit_price = trade['profit_exit_price'] if close_price >= trade['profit_exit_price'] else trade['stop_loss_price']
                profit = (exit_price - trade['entry_price']) * (trade['trade_size'] / trade['entry_price'])
                if profit < 0:
                    total_loss += abs(profit)
                balance += trade['trade_size'] + profit
                trade_data.append((trade['entry_date'], data.index[i], trade['entry_price'], exit_price, profit))
            else:
                new_active_trades.append(trade)
        active_trades = new_active_trades

    estimated_balance = balance
    for trade in active_trades:
        close_price = data['Close'].iloc[-1]
        unrealized_profit = (close_price - trade['entry_price']) * (trade['trade_size'] / trade['entry_price'])
        estimated_balance += trade['trade_size'] + unrealized_profit

    trade_df = pd.DataFrame(trade_data, columns=['Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Profit'])
    return data, trade_df, balance, total_loss, estimated_balance

# Execute trades with pyramiding
data, trade_df, ending_balance, total_loss, estimated_balance = execute_trades_with_pyramiding(data, profit_exit, loss_stop, starting_balance, max_pyramids, equity_percent)

# Calculate performance metrics
total_trades = len(trade_df)
profitable_trades = (trade_df['Profit'] > 0).sum()
total_profit = trade_df['Profit'].sum()
percent_profitable = profitable_trades / total_trades * 100 if total_trades > 0 else 0
profit_factor = trade_df['Profit'][trade_df['Profit'] > 0].sum() / abs(trade_df['Profit'][trade_df['Profit'] < 0].sum()) if total_trades > 0 else 0
max_drawdown = trade_df['Profit'].cumsum().min()

# Create performance summary
performance_summary = {
    'Total Trades': total_trades,
    'Percent Profitable': percent_profitable,
    'Total Profit (USD)': total_profit,
    'Total Loss (USD)': total_loss,
    'Profit Factor': profit_factor,
    'Max Drawdown (USD)': max_drawdown,
    'Starting Balance (USD)': starting_balance,
    'Ending Balance (USD)': ending_balance,
    'Estimated Ending Balance (USD)': estimated_balance
}

# Save trades to file
trades_file = os.path.join(output_path, f'trades_{timestamp}.csv')
trade_df.to_csv(trades_file, index=False)
print(f'Trades saved to {trades_file}')

# Save performance summary to file
performance_file = os.path.join(output_path, f'performance_summary_{timestamp}.csv')
pd.DataFrame([performance_summary]).to_csv(performance_file, index=False)

# Print performance summary
print("Performance Summary:")
for key, value in performance_summary.items():
    print(f"{key}: {value}")

print(f"Performance summary saved to {performance_file}")
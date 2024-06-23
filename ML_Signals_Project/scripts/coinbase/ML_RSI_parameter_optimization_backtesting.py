import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt

# Set the current working directory to the 'scripts' directory
script_cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_cwd)

# Determine the base directory of the project
base_dir = os.path.abspath(os.path.join(script_cwd, '..', '..'))

# Parameters
DATA_PATH = os.path.join(base_dir, 'data', 'coinbase', 'candles', 'BTC-USD', 'FIVE_MINUTE')
OUTPUT_BASE_PATH = os.path.join(script_cwd, 'test_trade_output_files')
START_DATE = datetime(2024, 5, 1)
END_DATE = datetime(2024, 6, 5)

# RSI Parameters
RSI_WINDOWS = list(range(10, 12))  # RSI window sizes from 10 to 14
RSI_OVERBOUGHT_RANGE = list(range(79, 88))  # Overbought levels from 80 to 87
RSI_OVERSOLD_RANGE = list(range(18, 26))  # Oversold levels from 23 to 28
RSI_PROFIT_EXIT_RANGE = [round(x, 2) for x in np.arange(0.03, 0.08, 0.01)]  # Profit exit levels from 0.03 to 0.07
RSI_LOSS_STOP_RANGE = [round(x, 2) for x in np.arange(0.01, 0.05, 0.01)]  # Loss stop levels from 0.01 to 0.04

# Trading Parameters
STARTING_BALANCE = 1000
MAX_PYRAMIDS = 1
EQUITY_PERCENT_RSI = 1.0

# Ensure output directory exists
output_path = os.path.join(OUTPUT_BASE_PATH, f'{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}')
os.makedirs(output_path, exist_ok=True)
log_file = os.path.join(output_path, 'rsi_tuning_log.txt')

def log_print(*args):
    with open(log_file, 'a') as f:
        print(*args, file=f)
    print(*args)

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
            log_print(f"Skipping file {file}: {e}")
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
        combined_data = combined_data[(combined_data.index >= start_date) & (combined_data.index <= end_date)]
        combined_data = combined_data.sort_index()
        return combined_data
    else:
        return pd.DataFrame(columns=['Start', 'Low', 'High', 'Open', 'Close', 'Volume'])

# Load the data
data = load_data_files(DATA_PATH, START_DATE, END_DATE)

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# Function to generate RSI signals
def generate_rsi_signals(data, rsi_overbought, rsi_oversold):
    data['RSI_Signal'] = 0
    data.loc[data['RSI'] < rsi_oversold, 'RSI_Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > rsi_overbought, 'RSI_Signal'] = -1  # Sell signal
    return data

# Function to execute trades
def execute_trades(data, profit_exit, loss_stop):
    balance = STARTING_BALANCE
    total_loss = 0
    active_trades = []
    trade_data = []

    for i in range(1, len(data)):
        signal = data['RSI_Signal'].iloc[i]

        if signal == 1 and len(active_trades) < MAX_PYRAMIDS:
            # Enter a new trade
            entry_price = data['Close'].iloc[i]
            trade_size = balance * EQUITY_PERCENT_RSI
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
    return trade_df, balance, total_loss, estimated_balance

# Placeholder for best parameters
best_params_rsi = None
best_score_rsi = -np.inf

# Perform Grid Search for RSI
for rsi_window in RSI_WINDOWS:
    for rsi_overbought in RSI_OVERBOUGHT_RANGE:
        for rsi_oversold in RSI_OVERSOLD_RANGE:
            for rsi_profit_exit in RSI_PROFIT_EXIT_RANGE:
                for rsi_loss_stop in RSI_LOSS_STOP_RANGE:
                    data = load_data_files(DATA_PATH, START_DATE, END_DATE)
                    calculate_rsi(data, rsi_window)
                    data = generate_rsi_signals(data, rsi_overbought, rsi_oversold)

                    # Execute trades with current parameters
                    trade_df, ending_balance, total_loss, estimated_balance = execute_trades(data, rsi_profit_exit, rsi_loss_stop)

                    # Calculate performance metrics
                    total_trades = len(trade_df)
                    profitable_trades = (trade_df['Profit'] > 0).sum()
                    total_profit = trade_df['Profit'].sum()
                    percent_profitable = profitable_trades / total_trades * 100 if total_trades > 0 else 0
                    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                    max_drawdown = trade_df['Profit'].cumsum().min()

                    # Log the results
                    log_print(f"RSI Window: {rsi_window}, Overbought: {rsi_overbought}, Oversold: {rsi_oversold}, Profit Exit: {rsi_profit_exit}, Loss Stop: {rsi_loss_stop}")
                    log_print(f"Total Trades: {total_trades}, Percent Profitable: {percent_profitable}, Total Profit: {total_profit}, Total Loss: {total_loss}, Profit Factor: {profit_factor}, Max Drawdown: {max_drawdown}, Ending Balance: {ending_balance}, Estimated Ending Balance: {estimated_balance}")

                    # Update best parameters if current score is better
                    if total_profit > best_score_rsi:
                        best_score_rsi = total_profit
                        best_params_rsi = (rsi_window, rsi_overbought, rsi_oversold, rsi_profit_exit, rsi_loss_stop)

# Best parameters found from Grid Search for RSI
log_print(f"Best RSI Parameters: Window: {best_params_rsi[0]}, Overbought: {best_params_rsi[1]}, Oversold: {best_params_rsi[2]}, Profit Exit: {best_params_rsi[3]}, Loss Stop: {best_params_rsi[4]}")
log_print(f"Best RSI Profit: {best_score_rsi}")

# Save the best parameters and performance summary
performance_summary_rsi = {
    'Best Parameters': best_params_rsi,
    'Best Profit': best_score_rsi
}
performance_file_rsi = os.path.join(output_path, 'performance_summary_rsi.csv')
pd.DataFrame([performance_summary_rsi]).to_csv(performance_file_rsi, index=False)
log_print(f'RSI Performance summary saved to {performance_file_rsi}')

# Retrain the model with the best RSI parameters
best_rsi_window, best_rsi_overbought, best_rsi_oversold, best_rsi_profit_exit, best_rsi_loss_stop = best_params_rsi
data = load_data_files(DATA_PATH, START_DATE, END_DATE)
calculate_rsi(data, best_rsi_window)
data = generate_rsi_signals(data, best_rsi_overbought, best_rsi_oversold)

# Split data into training and testing sets
X_rsi = data[['Close', 'RSI']]
y_rsi = data['RSI_Signal']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_rsi = imputer.fit_transform(X_rsi)

X_train_rsi, X_test_rsi, y_train_rsi, y_test_rsi = train_test_split(X_rsi, y_rsi, test_size=0.2, random_state=42)
best_model_rsi = RandomForestClassifier(random_state=42)
best_model_rsi.fit(X_train_rsi, y_train_rsi)

# Evaluate the best RSI model
y_pred_rsi = best_model_rsi.predict(X_test_rsi)
log_print("Best RSI Model Performance:")
log_print(classification_report(y_test_rsi, y_pred_rsi))
log_print(f'RSI Accuracy: {accuracy_score(y_test_rsi, y_pred_rsi)}')

# Save the best RSI model
model_file_rsi = os.path.join(output_path, 'random_forest_model_rsi.joblib')
joblib.dump(best_model_rsi, model_file_rsi)
log_print(f'Best RSI Model saved to {model_file_rsi}')

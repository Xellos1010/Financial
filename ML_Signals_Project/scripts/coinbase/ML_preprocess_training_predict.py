import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Set the current working directory to the 'scripts' directory
script_cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_cwd)
print(f"Current working directory set to: {script_cwd}")

# Determine the base directory of the project
base_dir = os.path.abspath(os.path.join(script_cwd, '..', '..'))
print(f"Base directory determined as: {base_dir}")

# Parameters
DATA_PATH = os.path.join(base_dir, 'data', 'coinbase', 'candles', 'BTC-USD', 'FIVE_MINUTE')
OUTPUT_BASE_PATH = os.path.join(script_cwd, 'test_trade_output_files')
START_DATE = datetime(2024, 5, 1)
END_DATE = datetime(2024, 6, 5)

# RSI Parameters
RSI_WINDOW = 12
RSI_OVERBOUGHT = 79
RSI_OVERSOLD = 26
RSI_PROFIT_EXIT = 0.05
RSI_LOSS_STOP = 0.02

# Bollinger Bands Parameters
VWMA_WINDOW = 20
FIB_LEVELS = [0.618, 1.0, 1.618, 2.618, 3.618, 4.618]
BB_PROFIT_EXIT = 0.05
BB_LOSS_STOP = 0.02

# Trading Parameters
STARTING_BALANCE = 1000
MAX_PYRAMIDS = 1
EQUITY_PERCENT_RSI = 0.65
EQUITY_PERCENT_BB = 0.45

# Ensure output directory exists
output_path = os.path.join(OUTPUT_BASE_PATH, f'{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}')
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
data = load_data_files(DATA_PATH, START_DATE, END_DATE)

# Function to calculate technical indicators (RSI, VWMA, and Fibonacci Bollinger Bands)
def calculate_technical_indicators(df, rsi_window, vwma_window, fib_levels):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate VWMA
    vwma_column = f'VWMA{vwma_window}'
    df[vwma_column] = (df['Close'] * df['Volume']).rolling(window=vwma_window).sum() / df['Volume'].rolling(window=vwma_window).sum()

    # Calculate Fibonacci Bollinger Bands based on VWMA
    df['BB_Mid'] = df[vwma_column]
    df['BB_Upper'] = df[vwma_column] + 2 * df[vwma_column].rolling(window=vwma_window).std()
    df['BB_Lower'] = df[vwma_column] - 2 * df[vwma_column].rolling(window=vwma_window).std()
    for level in fib_levels:
        df[f'Fib_Upper_{int(level*100)}'] = df['BB_Upper'] + (df['BB_Upper'] - df['BB_Lower']) * level
        df[f'Fib_Lower_{int(level*100)}'] = df['BB_Lower'] - (df['BB_Upper'] - df['BB_Lower']) * level

    df.dropna(inplace=True)
    return df

# Check if technical indicators file already exists
tech_indicators_file = os.path.join(output_path, f'technical_indicators_{RSI_WINDOW}_{VWMA_WINDOW}.csv')
if not os.path.exists(tech_indicators_file):
    # Calculate technical indicators
    data = calculate_technical_indicators(data, RSI_WINDOW, VWMA_WINDOW, FIB_LEVELS)

    # Save technical indicators and window to file
    data.to_csv(tech_indicators_file)
    print(f'Technical indicators saved to {tech_indicators_file}')
else:
    data = pd.read_csv(tech_indicators_file, index_col='Date', parse_dates=True)
    print(f'Technical indicators file {tech_indicators_file} already exists. Loading data...')

# Generate signals for RSI and Bollinger Bands strategies
def generate_signals(data, rsi_overbought, rsi_oversold, fib_levels):
    data['RSI_Signal'] = 0
    data.loc[data['RSI'] < rsi_oversold, 'RSI_Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > rsi_overbought, 'RSI_Signal'] = -1  # Sell signal
    
    data['BB_Signal'] = 0
    for level in fib_levels:
        data.loc[data['Close'] < data[f'Fib_Lower_{int(level*100)}'], 'BB_Signal'] = 1  # Buy signal
        data.loc[data['Close'] > data[f'Fib_Upper_{int(level*100)}'], 'BB_Signal'] = -1  # Sell signal
    return data

data = generate_signals(data, RSI_OVERBOUGHT, RSI_OVERSOLD, FIB_LEVELS)

# Define features and targets for RSI and BB strategies
features = ['Close', 'RSI', f'VWMA{VWMA_WINDOW}', 'BB_Mid', 'BB_Upper', 'BB_Lower'] + [f'Fib_Upper_{int(level*100)}' for level in FIB_LEVELS] + [f'Fib_Lower_{int(level*100)}' for level in FIB_LEVELS]
rsi_target = 'RSI_Signal'
bb_target = 'BB_Signal'

# Split data into training and testing sets for RSI
X_rsi = data[features]
y_rsi = data[rsi_target]
X_train_rsi, X_test_rsi, y_train_rsi, y_test_rsi = train_test_split(X_rsi, y_rsi, test_size=0.2, random_state=42)

# Split data into training and testing sets for BB
X_bb = data[features]
y_bb = data[bb_target]
X_train_bb, X_test_bb, y_train_bb, y_test_bb = train_test_split(X_bb, y_bb, test_size=0.2, random_state=42)

# Train a Random Forest Classifier for RSI with Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rsi = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_rsi.fit(X_train_rsi, y_train_rsi)

# Best model from grid search for RSI
best_model_rsi = grid_search_rsi.best_estimator_

# Evaluate the RSI model
y_pred_rsi = best_model_rsi.predict(X_test_rsi)
print("RSI Model Performance:")
print(classification_report(y_test_rsi, y_pred_rsi))
print(f'Accuracy: {accuracy_score(y_test_rsi, y_pred_rsi)}')

# Train a Random Forest Classifier for BB with Grid Search for hyperparameter tuning
grid_search_bb = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_bb.fit(X_train_bb, y_train_bb)

# Best model from grid search for BB
best_model_bb = grid_search_bb.best_estimator_

# Evaluate the BB model
y_pred_bb = best_model_bb.predict(X_test_bb)
print("BB Model Performance:")
print(classification_report(y_test_bb, y_pred_bb))
print(f'Accuracy: {accuracy_score(y_test_bb, y_pred_bb)}')

# Save the trained models
model_file_rsi = os.path.join(output_path, 'random_forest_model_rsi.joblib')
model_file_bb = os.path.join(output_path, 'random_forest_model_bb.joblib')
joblib.dump(best_model_rsi, model_file_rsi)
joblib.dump(best_model_bb, model_file_bb)
print(f'RSI Model saved to {model_file_rsi}')
print(f'BB Model saved to {model_file_bb}')

# Load the trained models
model_rsi = joblib.load(model_file_rsi)
model_bb = joblib.load(model_file_bb)

# Use the models to make predictions
data['Predicted_RSI_Signal'] = model_rsi.predict(data[features])
data['Predicted_BB_Signal'] = model_bb.predict(data[features])

# Function to execute trades with pyramiding
def execute_trades_with_pyramiding(data, profit_exit, loss_stop, starting_balance, max_pyramids, equity_percent, signal_column):
    balance = starting_balance
    total_loss = 0
    active_trades = []
    trade_data = []

    for i in range(1, len(data)):
        signal = data[signal_column].iloc[i]

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
    return trade_df, balance, total_loss, estimated_balance

# Execute trades with pyramiding for both strategies
trade_df_rsi, ending_balance_rsi, total_loss_rsi, estimated_balance_rsi = execute_trades_with_pyramiding(data, RSI_PROFIT_EXIT, RSI_LOSS_STOP, STARTING_BALANCE * EQUITY_PERCENT_RSI, MAX_PYRAMIDS, EQUITY_PERCENT_RSI, 'Predicted_RSI_Signal')
trade_df_bb, ending_balance_bb, total_loss_bb, estimated_balance_bb = execute_trades_with_pyramiding(data, BB_PROFIT_EXIT, BB_LOSS_STOP, STARTING_BALANCE * EQUITY_PERCENT_BB, MAX_PYRAMIDS, EQUITY_PERCENT_BB, 'Predicted_BB_Signal')

# Calculate performance metrics for RSI strategy
total_trades_rsi = len(trade_df_rsi)
profitable_trades_rsi = (trade_df_rsi['Profit'] > 0).sum()
total_profit_rsi = trade_df_rsi['Profit'].sum()
percent_profitable_rsi = profitable_trades_rsi / total_trades_rsi * 100 if total_trades_rsi > 0 else 0
profit_factor_rsi = trade_df_rsi['Profit'][trade_df_rsi['Profit'] > 0].sum() / abs(trade_df_rsi['Profit'][trade_df_rsi['Profit'] < 0].sum()) if total_trades_rsi > 0 else 0
max_drawdown_rsi = trade_df_rsi['Profit'].cumsum().min()

# Calculate performance metrics for BB strategy
total_trades_bb = len(trade_df_bb)
profitable_trades_bb = (trade_df_bb['Profit'] > 0).sum()
total_profit_bb = trade_df_bb['Profit'].sum()
percent_profitable_bb = profitable_trades_bb / total_trades_bb * 100 if total_trades_bb > 0 else 0
profit_factor_bb = trade_df_bb['Profit'][trade_df_bb['Profit'] > 0].sum() / abs(trade_df_bb['Profit'][trade_df_bb['Profit'] < 0].sum()) if total_trades_bb > 0 else 0
max_drawdown_bb = trade_df_bb['Profit'].cumsum().min()

# Create performance summary for RSI strategy
performance_summary_rsi = {
    'Total Trades': total_trades_rsi,
    'Percent Profitable': percent_profitable_rsi,
    'Total Profit (USD)': total_profit_rsi,
    'Total Loss (USD)': total_loss_rsi,
    'Profit Factor': profit_factor_rsi,
    'Max Drawdown (USD)': max_drawdown_rsi,
    'Starting Balance (USD)': STARTING_BALANCE * EQUITY_PERCENT_RSI,
    'Ending Balance (USD)': ending_balance_rsi,
    'Estimated Ending Balance (USD)': estimated_balance_rsi
}

# Create performance summary for BB strategy
performance_summary_bb = {
    'Total Trades': total_trades_bb,
    'Percent Profitable': percent_profitable_bb,
    'Total Profit (USD)': total_profit_bb,
    'Total Loss (USD)': total_loss_bb,
    'Profit Factor': profit_factor_bb,
    'Max Drawdown (USD)': max_drawdown_bb,
    'Starting Balance (USD)': STARTING_BALANCE * EQUITY_PERCENT_BB,
    'Ending Balance (USD)': ending_balance_bb,
    'Estimated Ending Balance (USD)': estimated_balance_bb
}

# Save trades to file for both strategies
trades_file_rsi = os.path.join(output_path, 'trades_rsi.csv')
trades_file_bb = os.path.join(output_path, 'trades_bb.csv')
trade_df_rsi.to_csv(trades_file_rsi, index=False)
trade_df_bb.to_csv(trades_file_bb, index=False)
print(f'Trades for RSI saved to {trades_file_rsi}')
print(f'Trades for BB saved to {trades_file_bb}')

# Save performance summary to file for both strategies
performance_file_rsi = os.path.join(output_path, 'performance_summary_rsi.csv')
performance_file_bb = os.path.join(output_path, 'performance_summary_bb.csv')
pd.DataFrame([performance_summary_rsi]).to_csv(performance_file_rsi, index=False)
pd.DataFrame([performance_summary_bb]).to_csv(performance_file_bb, index=False)

# Print performance summary for both strategies
print("RSI Performance Summary:")
for key, value in performance_summary_rsi.items():
    print(f"{key}: {value}")

print("BB Performance Summary:")
for key, value in performance_summary_bb.items():
    print(f"{key}: {value}")

print(f"Performance summary for RSI saved to {performance_file_rsi}")
print(f"Performance summary for BB saved to {performance_file_bb}")

# Plot the data with signals and trades
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data[f'VWMA{VWMA_WINDOW}'], label=f'VWMA{VWMA_WINDOW}')
plt.plot(data.index, data['BB_Mid'], label='VWMA Mid')
plt.plot(data.index, data['BB_Upper'], label='BB Upper')
plt.plot(data.index, data['BB_Lower'], label='BB Lower')
for level in FIB_LEVELS:
    plt.plot(data.index, data[f'Fib_Upper_{int(level*100)}'], label=f'Fib Upper {int(level*100)}')
    plt.plot(data.index, data[f'Fib_Lower_{int(level*100)}'], label=f'Fib Lower {int(level*100)}')

# Plot buy and sell signals for RSI strategy
buy_signals_rsi = data[data['Predicted_RSI_Signal'] == 1]
sell_signals_rsi = data[data['Predicted_RSI_Signal'] == -1]
plt.scatter(buy_signals_rsi.index, buy_signals_rsi['Close'], marker='^', color='green', label='RSI Buy Signal', alpha=1)
plt.scatter(sell_signals_rsi.index, sell_signals_rsi['Close'], marker='v', color='red', label='RSI Sell Signal', alpha=1)

# Plot buy and sell signals for BB strategy
buy_signals_bb = data[data['Predicted_BB_Signal'] == 1]
sell_signals_bb = data[data['Predicted_BB_Signal'] == -1]
plt.scatter(buy_signals_bb.index, buy_signals_bb['Close'], marker='o', color='blue', label='BB Buy Signal', alpha=0.5)
plt.scatter(sell_signals_bb.index, sell_signals_bb['Close'], marker='x', color='purple', label='BB Sell Signal', alpha=0.5)

plt.title('Trading Signals with RSI and Bollinger Bands Strategies')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()

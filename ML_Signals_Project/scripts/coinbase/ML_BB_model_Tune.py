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

# Bollinger Bands Parameters
VWMA_WINDOWS = list(range(15, 26))  # VWMA window sizes from 15 to 25
FIB_LEVELS = [0.618, 1.0, 1.618, 2.618, 3.618, 4.618]
BB_PROFIT_EXIT_RANGE = [round(x, 2) for x in np.arange(0.03, 0.12, 0.01)]  # Profit exit levels from 0.03 to 0.11
BB_LOSS_STOP_RANGE = [round(x, 2) for x in np.arange(0.01, 0.05, 0.01)]  # Loss stop levels from 0.01 to 0.04

# Trading Parameters
STARTING_BALANCE = 1000
MAX_PYRAMIDS = 1
EQUITY_PERCENT_BB = 0.45

# Ensure output directory exists
output_path = os.path.join(OUTPUT_BASE_PATH, f'{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}')
os.makedirs(output_path, exist_ok=True)
log_file = os.path.join(output_path, 'bb_tuning_log.txt')

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

# Function to calculate VWMA
def calculate_vwma(data, window):
    vwma_column = f'VWMA{window}'
    data[vwma_column] = (data['Close'] * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return data

# Function to calculate Fibonacci Bollinger Bands
def calculate_fib_bollinger_bands(data, window, fib_levels):
    vwma_column = f'VWMA{window}'
    data['BB_Mid'] = data[vwma_column]
    data['BB_Upper'] = data[vwma_column] + 2 * data[vwma_column].rolling(window=window).std()
    data['BB_Lower'] = data[vwma_column] - 2 * data[vwma_column].rolling(window=window).std()
    for level in fib_levels:
        data[f'Fib_Upper_{int(level*100)}'] = data['BB_Upper'] + (data['BB_Upper'] - data['BB_Lower']) * level
        data[f'Fib_Lower_{int(level*100)}'] = data['BB_Lower'] - (data['BB_Upper'] - data['BB_Lower']) * level
    return data

# Function to generate BB signals
def generate_bb_signals(data, fib_levels):
    data['BB_Signal'] = 0
    for level in fib_levels:
        data.loc[data['Close'] < data[f'Fib_Lower_{int(level*100)}'], 'BB_Signal'] = 1  # Buy signal
        data.loc[data['Close'] > data[f'Fib_Upper_{int(level*100)}'], 'BB_Signal'] = -1  # Sell signal
    return data

# Placeholder for best parameters
best_params_bb = None
best_score_bb = -np.inf

# Perform Grid Search for Bollinger Bands
for vwma_window in VWMA_WINDOWS:
    for bb_profit_exit in BB_PROFIT_EXIT_RANGE:
        for bb_loss_stop in BB_LOSS_STOP_RANGE:
            data = load_data_files(DATA_PATH, START_DATE, END_DATE)
            data = calculate_vwma(data, vwma_window)
            data = calculate_fib_bollinger_bands(data, vwma_window, FIB_LEVELS)
            data = generate_bb_signals(data, FIB_LEVELS)

            # Define features and target for the current VWMA window
            bb_features = ['Close', f'VWMA{vwma_window}', 'BB_Mid', 'BB_Upper', 'BB_Lower'] + [f'Fib_Upper_{int(level*100)}' for level in FIB_LEVELS] + [f'Fib_Lower_{int(level*100)}' for level in FIB_LEVELS]
            X_bb = data[bb_features]
            y_bb = data['BB_Signal']

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X_bb = imputer.fit_transform(X_bb)

            X_train_bb, X_test_bb, y_train_bb, y_test_bb = train_test_split(X_bb, y_bb, test_size=0.2, random_state=42)
            
            # Train a Random Forest Classifier
            model_bb = RandomForestClassifier(random_state=42)
            model_bb.fit(X_train_bb, y_train_bb)
            
            # Evaluate the BB model
            y_pred_bb = model_bb.predict(X_test_bb)
            score = accuracy_score(y_test_bb, y_pred_bb)
            
            log_print(f"Evaluating VWMA Window: {vwma_window}, BB Profit Exit: {bb_profit_exit}, BB Loss Stop: {bb_loss_stop}, Score: {score}")
            
            # Update best parameters if current score is better
            if score > best_score_bb:
                best_score_bb = score
                best_params_bb = (vwma_window, bb_profit_exit, bb_loss_stop)

# Best parameters found from Grid Search for Bollinger Bands
log_print(f"Best BB Parameters: VWMA Window: {best_params_bb[0]}, BB Profit Exit: {best_params_bb[1]}, BB Loss Stop: {best_params_bb[2]}")
log_print(f"Best BB Accuracy Score: {best_score_bb}")

# Retrain the model with the best Bollinger Bands parameters
best_vwma_window, best_bb_profit_exit, best_bb_loss_stop = best_params_bb
data = load_data_files(DATA_PATH, START_DATE, END_DATE)
data = calculate_vwma(data, best_vwma_window)
data = calculate_fib_bollinger_bands(data, best_vwma_window, FIB_LEVELS)
data = generate_bb_signals(data, FIB_LEVELS)
bb_features = ['Close', f'VWMA{best_vwma_window}', 'BB_Mid', 'BB_Upper', 'BB_Lower'] + [f'Fib_Upper_{int(level*100)}' for level in FIB_LEVELS] + [f'Fib_Lower_{int(level*100)}' for level in FIB_LEVELS]
X_bb = data[bb_features]
y_bb = data['BB_Signal']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_bb = imputer.fit_transform(X_bb)

X_train_bb, X_test_bb, y_train_bb, y_test_bb = train_test_split(X_bb, y_bb, test_size=0.2, random_state=42)
best_model_bb = RandomForestClassifier(random_state=42)
best_model_bb.fit(X_train_bb, y_train_bb)

# Evaluate the best BB model
y_pred_bb = best_model_bb.predict(X_test_bb)
log_print("Best BB Model Performance:")
log_print(classification_report(y_test_bb, y_pred_bb))
log_print(f'BB Accuracy: {accuracy_score(y_test_bb, y_pred_bb)}')

# Save the best BB model
model_file_bb = os.path.join(output_path, 'random_forest_model_bb.joblib')
joblib.dump(best_model_bb, model_file_bb)
log_print(f'Best BB Model saved to {model_file_bb}')

# Save performance summary to file
performance_summary_bb = {
    'Best Parameters': best_params_bb,
    'Best Score': best_score_bb,
    'Accuracy': accuracy_score(y_test_bb, y_pred_bb)
}
performance_file_bb = os.path.join(output_path, 'performance_summary_bb.csv')
pd.DataFrame([performance_summary_bb]).to_csv(performance_file_bb, index=False)
log_print(f'BB Performance summary saved to {performance_file_bb}')

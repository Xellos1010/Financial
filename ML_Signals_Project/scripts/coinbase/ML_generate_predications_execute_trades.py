import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

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
output_path = os.path.join(output_base_path, f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}')
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

# Function to calculate technical indicators (RSI, moving averages, etc.)
def calculate_technical_indicators(df, rsi_window):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Calculate Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(window=20).std()

    df.dropna(inplace=True)
    return df

# Check if RSI file already exists
rsi_file = os.path.join(output_path, f'RSI_values_{rsi_window}.csv')
if not os.path.exists(rsi_file):
    # Calculate technical indicators
    data = calculate_technical_indicators(data, rsi_window)

    # Save RSI values and window to file
    data.to_csv(rsi_file)
    print(f'RSI values saved to {rsi_file}')
else:
    data = pd.read_csv(rsi_file, index_col='Date', parse_dates=True)
    print(f'RSI file {rsi_file} already exists. Loading data...')
    
    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define parameters for signals
rsi_overbought = 79
rsi_oversold = 26

# Function to generate signals
def generate_signals(data, rsi_overbought, rsi_oversold):
    data['Signal'] = 0
    data.loc[data['RSI'] < rsi_oversold, 'Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > rsi_overbought, 'Signal'] = -1  # Sell signal
    return data

# Generate signals
data = generate_signals(data, rsi_overbought, rsi_oversold)

# Define features and target
features = ['Close', 'RSI', 'MA50', 'MA200', 'BB_Mid', 'BB_Upper', 'BB_Lower']
target = 'Signal'

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the trained model
import joblib
model_file = os.path.join(output_path, 'random_forest_model.joblib')
joblib.dump(model, model_file)
print(f'Model saved to {model_file}')



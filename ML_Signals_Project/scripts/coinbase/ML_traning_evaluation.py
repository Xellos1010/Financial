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

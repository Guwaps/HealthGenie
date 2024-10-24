# forecasting.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Function to train models for systolic and diastolic predictions
def train_models(timestamps, systolic_values, diastolic_values):
    # Prepare DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'systolic': systolic_values,
        'diastolic': diastolic_values
    })
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10 ** 9

    # Train systolic model
    X = df[['timestamp']]
    y_systolic = df['systolic']
    systolic_model = LinearRegression()
    systolic_model.fit(X, y_systolic)
    joblib.dump(systolic_model, 'systolic_model.pkl')

    # Train diastolic model
    y_diastolic = df['diastolic']
    diastolic_model = LinearRegression()
    diastolic_model.fit(X, y_diastolic)
    joblib.dump(diastolic_model, 'diastolic_model.pkl')

    return "Models trained successfully"

# Function to predict future systolic and diastolic values
def predict_future(future_timestamps):
    # Load models
    systolic_model = joblib.load('systolic_model.pkl')
    diastolic_model = joblib.load('diastolic_model.pkl')

    future_data = pd.DataFrame({
        'timestamp': future_timestamps,
        # Add other relevant fields if necessary
    })

    # Convert to datetime and then to integer (e.g., in seconds)
    future_data['timestamp'] = pd.to_datetime(future_data['timestamp']).astype('int64') // 10 ** 9  # Convert to seconds

    # Predict systolic and diastolic values
    predicted_systolic = systolic_model.predict(future_data[['timestamp']])
    predicted_diastolic = diastolic_model.predict(future_data[['timestamp']])

    return predicted_systolic.tolist(), predicted_diastolic.tolist()

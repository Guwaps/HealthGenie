import datetime
from flask import Flask, request, jsonify
import re
import joblib
import pandas as pd
from ChatBotAPI import chatWithBot
from forecasting_healthData import train_models, predict_future

app = Flask(__name__)

# Function to generate future timestamps
def generate_future_timestamps(period, units):
    future_timestamps = []
    if period == 'days':
        for i in range(1, units + 1):
            future_timestamps.append((datetime.datetime.now() + datetime.timedelta(days=i)).timestamp() * 1000)
    elif period == 'months':
        for i in range(1, units + 1):
            future_date = (datetime.datetime.now() + datetime.timedelta(weeks=4 * i))
            future_timestamps.append(future_date.timestamp() * 1000)
    return future_timestamps


@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    print("Received data for training:", data)  # Print the received data for training
    timestamps = data.get('timestamps')
    systolic_values = data.get('systolic')
    diastolic_values = data.get('diastolic')

    if timestamps and systolic_values and diastolic_values:
        status = train_models(timestamps, systolic_values, diastolic_values)
        return jsonify({'status': status})
    else:
        return jsonify({'error': 'Invalid input data'}), 400


@app.route('/predict', methods=['POST'])
def predict_bp():
    data = request.get_json()
    print("Received data for prediction:", data)  # Print the received data for prediction
    period = data.get('period', 'days')
    units = data.get('units', 1)

    future_timestamps = generate_future_timestamps(period, units)

    if future_timestamps:
        systolic, diastolic = predict_future(future_timestamps)

        # Extract the first element since all values are the same
        first_systolic = [systolic[0]] if systolic else []
        first_diastolic = [diastolic[0]] if diastolic else []

        print(f"Predicted Systolic: {first_systolic}, Predicted Diastolic: {first_diastolic}")
        return jsonify({
            'predicted_systolic': first_systolic,
            'predicted_diastolic': first_diastolic
        })
    else:
        return jsonify({'error': 'No timestamps provided'}), 400


@app.route('/chat', methods=['POST'])
def chat_bot():
    data = request.get_json()
    print("Received chat input:", data)  # Print the received chat input
    chat_input = data.get('chatInput')

    if chat_input:
        match = re.search(r'(\d+)\s*/\s*(\d+)', chat_input)
        if match:
            systolic_BP = match.group(1)
            diastolic_BP = match.group(2)

            inputText = chat_input.replace(systolic_BP, "{Systolic_BP}").replace(diastolic_BP, "{Diastolic_BP}")
            response, tag = chatWithBot(inputText)
            response = response.replace("{Systolic_BP}", systolic_BP).replace("{Diastolic_BP}", diastolic_BP)
        else:
            response, tag = chatWithBot(chat_input)

        return jsonify({'chatBotReply': response, 'detectedTag': tag})
    else:
        return jsonify({'error': 'No input provided'}), 400


def classify_blood_pressure(systolic, diastolic):
    """Classify blood pressure based on systolic and diastolic values."""
    if systolic < 120 and diastolic < 80:
        return "Normal"  # Green
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"  # Yellow
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "High Blood Pressure (Hypertension) Stage 1"  # Orange
    elif systolic >= 140 or diastolic >= 90:
        return "High Blood Pressure (Hypertension) Stage 2"  # Brown
    elif systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis (consult your doctor immediately)"  # Red
    else:
        return "Normal"  # Default to normal if logic fails


@app.route('/predict_sickness', methods=['POST'])
def predict_custom():
    """Endpoint to predict blood pressure stage and possible sickness."""
    data = request.get_json()
    print("Received custom prediction input:", data)  # Debugging log

    systolic = data.get('systolic')
    diastolic = data.get('diastolic')

    if systolic is not None and diastolic is not None:
        # Load the sickness prediction model and mappings
        sickness_model = joblib.load('sickness_prediction_model.pkl')
        sickness_mapping = joblib.load('sickness_mapping.pkl')

        # Classify the blood pressure stage
        bp_stage = classify_blood_pressure(systolic, diastolic)

        if bp_stage == "Normal":
            possible_sickness = "No sickness detected"
        else:
            # Predict sickness (round to the nearest integer for classification)
            sickness_pred = sickness_model.predict([[systolic, diastolic]])[0]
            sickness_pred = int(round(sickness_pred))
            possible_sickness = sickness_mapping.get(sickness_pred, "Unknown Sickness")

        # Prepare response
        response = {
            'blood_pressure_stage': bp_stage,
            'possible_sickness': possible_sickness
        }

        print(f"Response: {response}")  # Debugging log
        return jsonify(response)
    else:
        return jsonify({'error': 'Systolic and Diastolic values must be provided'}), 400

@app.route('/')
def home():
    return "Welcome to HealthGinie!"

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is up and running!"

if __name__ == '__main__':
    app.run()

import datetime
from flask import Flask, request, jsonify
import re
import joblib
import numpy as np
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


# Function to classify blood pressure stage
def classify_blood_pressure(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "High Blood Pressure (Hypertension) Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "High Blood Pressure (Hypertension) Stage 2"
    elif systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis (consult your doctor immediately)"
    else:
        return "Normal"

@app.route('/predict_sickness', methods=['POST'])
def predict_custom():
    """Endpoint to predict blood pressure stage and multiple possible sicknesses."""
    data = request.get_json()
    print("Received input:", data)  # Debugging log

    systolic = data.get('systolic')
    diastolic = data.get('diastolic')

    if systolic is not None and diastolic is not None:
        try:
            # Load model and label encoder
            sickness_model = joblib.load('sickness/sickness_prediction_model.pkl')
            label_encoder = joblib.load('sickness/sickness_label_encoder.pkl')

            # Classify blood pressure stage
            bp_stage = classify_blood_pressure(systolic, diastolic)

            if bp_stage == "Normal":
                possible_sicknesses = ["No sickness detected"]
            else:
                # Predict probabilities for each sickness class
                sickness_probs = sickness_model.predict_proba([[systolic, diastolic]])[0]

                # Get top N sickness predictions (e.g., top 3)
                top_n = 3
                top_indices = np.argsort(sickness_probs)[::-1][:top_n]
                possible_sicknesses = label_encoder.inverse_transform(top_indices).tolist()

            # Prepare the response
            response = {
                'blood_pressure_stage': bp_stage,
                'possible_sicknesses': possible_sicknesses
            }

            print(f"Response: {response}")  # Debugging log
            return jsonify(response)

        except Exception as e:
            print(f"Error: {e}")  # Debugging log
            return jsonify({'error': 'Internal server error'}), 500
    else:
        return jsonify({'error': 'Systolic and Diastolic values must be provided'}), 400



@app.route('/')
def home():
    return "Welcome to HealthGinie!"

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is up and running!"

if __name__ == '__main__':
    app.run(host='192.168.210.236', port=5000)
    # app.run(debug=True)


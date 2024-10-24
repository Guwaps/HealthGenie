import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('large_hypertension_sickness_dataset.csv')

# Prepare the data for training
X = data[['Systolic_BP', 'Diastolic_BP']]
y_sickness = data['Possible_Sickness'].astype('category').cat.codes  # Encode sickness

# Train-test split
X_train, X_test, y_sick_train, y_sick_test = train_test_split(X, y_sickness, test_size=0.2, random_state=42)

# Train Linear Regression model for sickness prediction
sickness_model = LinearRegression()
sickness_model.fit(X_train, y_sick_train)

# Save the model and mappings
joblib.dump(sickness_model, 'sickness_prediction_model.pkl')

sickness_mapping = dict(enumerate(data['Possible_Sickness'].astype('category').cat.categories))
joblib.dump(sickness_mapping, 'sickness_mapping.pkl')

print("Model trained and saved!")

# Blood pressure stage classification logic
def classify_blood_pressure(systolic, diastolic):
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
        return "Normal"  # Green

# Custom input prediction function
def custom_input_prediction():
    # Load model and mappings for sickness prediction
    sickness_model = joblib.load('sickness_prediction_model.pkl')
    sickness_mapping = joblib.load('sickness_mapping.pkl')

    # Get custom inputs from user
    try:
        systolic = float(input("Enter Systolic Blood Pressure: "))
        diastolic = float(input("Enter Diastolic Blood Pressure: "))

        # Classify blood pressure stage
        bp_stage = classify_blood_pressure(systolic, diastolic)

        # If blood pressure is normal, skip sickness prediction
        if bp_stage == "Normal":
            possible_sickness = "No sickness detected"
        else:
            # Predict possible sickness (rounded to nearest integer)
            sickness_pred = sickness_model.predict([[systolic, diastolic]])[0]
            sickness_pred = int(round(sickness_pred))
            possible_sickness = sickness_mapping.get(sickness_pred, "Unknown Condition")

        # Display results
        print("\nPredictions:")
        print(f"Blood Pressure Stage: {bp_stage}")
        print(f"Possible Sickness: {possible_sickness}")

    except ValueError:
        print("Invalid input. Please enter numeric values for blood pressure.")

# Run the custom input prediction
if __name__ == '__main__':
    custom_input_prediction()

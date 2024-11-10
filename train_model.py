import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
data = pd.read_csv('assets/Hypertension-Related Health Metrics Dataset.csv')

# Prepare the data for training
X = data[['Systolic_BP', 'Diastolic_BP']]  # Features

# Encode sickness labels using LabelEncoder
label_encoder = LabelEncoder()
y_sickness = label_encoder.fit_transform(data['Possible_Sickness'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_sickness, test_size=0.2, random_state=42)

# Train a Logistic Regression model
sickness_model = LogisticRegression(max_iter=1000, multi_class='ovr')
sickness_model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(sickness_model, 'sickness/sickness_prediction_model.pkl')
joblib.dump(label_encoder, 'sickness/sickness_label_encoder.pkl')

print("Model trained and saved!")

# ---- Evaluation ----
# Predict on the test set
y_pred = sickness_model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:")
print(class_report)

# ---- Blood Pressure Classification Logic ----
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

# Custom input prediction function
def custom_input_prediction():
    # Load model and label encoder
    sickness_model = joblib.load('sickness/sickness_prediction_model.pkl')
    label_encoder = joblib.load('sickness/sickness_label_encoder.pkl')

    try:
        systolic = float(input("Enter Systolic Blood Pressure: "))
        diastolic = float(input("Enter Diastolic Blood Pressure: "))

        # Classify blood pressure stage
        bp_stage = classify_blood_pressure(systolic, diastolic)

        # If BP is normal, skip sickness prediction
        if bp_stage == "Normal":
            possible_sickness = "No sickness detected"
        else:
            # Predict sickness
            sickness_pred = sickness_model.predict([[systolic, diastolic]])[0]
            possible_sickness = label_encoder.inverse_transform([sickness_pred])[0]

        # Display results
        print("\nPredictions:")
        print(f"Blood Pressure Stage: {bp_stage}")
        print(f"Possible Sickness: {possible_sickness}")

    except ValueError:
        print("Invalid input. Please enter numeric values for blood pressure.")

# Run the custom input prediction
if __name__ == '__main__':
    custom_input_prediction()

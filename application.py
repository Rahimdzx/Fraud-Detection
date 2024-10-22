from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the scaler and model
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
model = pickle.load(open('model/fraud_detection_model.pkl', 'rb'))

# Route for the form, loading 'home.html'
@app.route("/")
def home():
    return render_template('home.html')

# Route to handle CSV file upload and make batch predictions
@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    # Read the uploaded CSV file
    data = pd.read_csv(file)
    
    # Check if all necessary columns are present
    expected_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    if not all(col in data.columns for col in expected_columns):
        return f"CSV must contain columns: {expected_columns}", 400
    
    # Extract feature values
    features = data[expected_columns].values
    
    # Scale the features using the scaler
    scaled_features = scaler.transform(features)
    
    # Make predictions for all rows
    predictions = model.predict(scaled_features)
    
    # Convert predictions to fraud/non-fraud labels
    data['Prediction'] = ['Fraud' if p == 1 else 'Non-Fraud' for p in predictions]
    
    # Convert the DataFrame to a format for rendering in HTML
    data_html = data[['Time', 'Amount', 'Prediction']].to_html(classes='data', header=True, index=False)

    # Render the results in a table format using the results.html template
    return render_template('results.html', table=data_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

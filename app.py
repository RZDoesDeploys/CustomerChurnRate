import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from catboost import CatBoostClassifier

# Add the 'project' directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

app = Flask(__name__)
CORS(app)

# CORS Headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Load the model once when the server starts
model = CatBoostClassifier()
model.load_model('model/nb1_catboost_model')

@app.route("/get_churn_rate", methods=['POST'])
def get_churn_rate():
    data = request.json

    # Define the expected input fields
    expected_fields = [
        'ClientPeriod', 'MonthlySpending', 'TotalSpent', 'Sex',
        'IsSeniorCitizen', 'HasPartner', 'HasChild', 'HasPhoneService',
        'HasMultiplePhoneNumbers', 'HasInternetService',
        'HasOnlineSecurityService', 'HasOnlineBackup', 'HasDeviceProtection',
        'HasTechSupportAccess', 'HasOnlineTV', 'HasMovieSubscription',
        'HasContractPhone', 'IsBillingPaperless', 'PaymentMethod'
    ]

    # Extract the input values in the correct order
    input_values = [data.get(field) for field in expected_fields]

    # Predict the probabilities using the model
    output = model.predict_proba([input_values])[0]

    # Create a response dictionary with the extracted inputs and the prediction
    response = {
        "NO_CHURN": output[0],
        "CHURN": output[1]
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000)

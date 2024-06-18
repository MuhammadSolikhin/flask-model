from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# Initialize Firestore
service_account_path = os.getenv('FIREBASE_CREDENTIALS')
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the trained model (adjust the path as needed)
model = tf.keras.models.load_model('Model.h5', compile=False)

def get_transactions_data():
    users_ref = db.collection('users')
    users = users_ref.stream()

    data = []
    for user in users:
        transactions_ref = users_ref.document(user.id).collection('transaction')
        transactions = transactions_ref.stream()
        for transaction in transactions:
            data.append(transaction.to_dict())

    return pd.DataFrame(data)

def calculate_growth(data):
    growth_per_month = {}
    for month in data["Month"].unique():
        monthly_data = data[data["Month"] == month]
        if not monthly_data.empty:
            start_value = monthly_data.iloc[0]["Cumulative"]
            end_value = monthly_data.iloc[-1]["Cumulative"]
            growth = ((end_value - start_value) / start_value) * 100
            growth_per_month[month] = growth
    return growth_per_month

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            token = token.split()[1]
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['GET'])
@token_required
def predict():
    data = get_transactions_data()
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
    data = data.sort_values(by="Date")

    data["Cumulative"] = data["income"] - data["expense"]
    data["Month"] = data["Date"].dt.month

    monthly_growth = calculate_growth(data)
    valid_months_growth = [growth for growth in monthly_growth.values() if not np.isnan(growth)]
    avg_growth = np.mean(valid_months_growth) if valid_months_growth else 0

    pred_395 = model.predict(np.array([[395]]))
    threshold = 2.50  # Change according to group decision

    start_value = data.iloc[-1]["Cumulative"]
    end_value = pred_395[0][0]  # Adjust indexing according to model output
    pred_growth = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0

    if pred_growth >= avg_growth + threshold:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangatlah baik!"
    elif pred_growth < avg_growth - threshold:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% cukup buruk ;-;"
    else:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangat stabil!"

    return jsonify({'prediction': pred_growth, 'growth_message': growth_message, 'monthly_growth': monthly_growth})

@app.route('/', methods=['GET'])
def get_message():
    return jsonify(message='Hey, your app is working')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# Initialize Firestore with JSON from environment variable
service_account_info = json.loads(os.getenv('FIREBASE_CREDENTIALS'))
# with open('firebase.json', 'r') as file:
#     service_account_info = json.load(file)

cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the trained model (adjust the path as needed)
model = tf.keras.models.load_model('Model.h5', compile=False)

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
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

def get_transactions_data(user_id):
    transactions_ref = db.collection('users').document(user_id).collection('transaction')
    transactions = transactions_ref.stream()
    data = []
    for transaction in transactions:
        trans_dict = transaction.to_dict()
        if 'date' in trans_dict:
            if trans_dict['date'] == firestore.SERVER_TIMESTAMP:
                trans_dict['date'] = datetime.now()
            else:
                trans_dict['date'] = trans_dict['date'].replace(tzinfo=None)
        data.append(trans_dict)
    return pd.DataFrame(data)

def calculate_growth(data):
    growth_per_month = {}
    for month in data["Month"].unique():
        monthly_data = data[data["Month"] == month]
        if not monthly_data.empty:
            start_value = monthly_data.iloc[0]["Cumulative"]
            end_value = monthly_data.iloc[-1]["Cumulative"]
            growth = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0
            growth_per_month[int(month)] = float(growth)  # Convert keys and values to standard Python types
    return growth_per_month

@app.route('/predict', methods=['GET'])
@token_required
def predict():
    user_id = request.user['uid']
    data = get_transactions_data(user_id)
    print(data.head())
    if 'date' not in data.columns:
        return jsonify({'message': 'Date column not found in data'}), 400

    data["date"] = pd.to_datetime(data["date"], errors='coerce')
    data = data.sort_values(by="date")

    # Calculate cumulative based on type (income/expense)
    data["Cumulative"] = data.apply(lambda x: x["amount"] if x["type"] == "income" else -x["amount"], axis=1).cumsum()
    data["Month"] = data["date"].dt.month

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

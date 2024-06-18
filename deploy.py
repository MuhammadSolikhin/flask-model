from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

# Load the trained model (adjust the path as needed)
model = tf.keras.models.load_model('Model.h5', compile=False)

# Load and preprocess the dataset
data = pd.read_csv("https://raw.githubusercontent.com/MosesSinanta/Repository_1/main/dataset_1.csv")
data["Date / Time"] = pd.to_datetime(data["Date / Time"], format="%d-%m-%Y")
data["Day"] = data["Date / Time"].dt.day
data["Month"] = data["Date / Time"].dt.month

# Add "Numerical Date" column based on day of year
data["Numerical Date"] = data["Date / Time"].dt.dayofyear
data = data.drop_duplicates(subset=["Numerical Date"], keep="last")

# Drop unnecessary columns
data = data.drop(columns=["Date / Time", "Mode", "Category", "Sub category", "Income / Expense", "Debit / Credit", "Day"])

# Features and target
X = data[["Numerical Date"]]
y = data["Cumulative"]

# Function to calculate growth
def calculate_growth(data):
    if data.empty:
        return 0
    start_value = data.iloc[0]["Cumulative"]
    end_value = data.iloc[-1]["Cumulative"]
    growth = ((end_value - start_value) / start_value) * 100
    return growth

# Calculate growth for specific months
oct_growth = calculate_growth(data[data["Month"] == 10])
nov_growth = calculate_growth(data[data["Month"] == 11])
dec_growth = calculate_growth(data[data["Month"] == 12])
avg_growth = np.mean([oct_growth, nov_growth, dec_growth])

@app.route('/predict', methods=['GET'])
def predict():
    # Predict for the input value 395
    pred_395 = model.predict(np.array([[395]]))
    threshold = 2.50  # Threshold for growth comparison

    # Calculate growth based on prediction
    start_value = data.iloc[-1]["Cumulative"]
    end_value = pred_395[0][0]  # Extract the prediction value from the array
    pred_growth = ((end_value - start_value) / start_value) * 100

    # Determine growth message
    if pred_growth >= avg_growth + threshold:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangatlah baik!"
    elif pred_growth < avg_growth - threshold:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% cukup buruk ;-;"
    else:
        growth_message = f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangat stabil!"

    return jsonify({'prediction': pred_growth, 'growth_message': growth_message})

@app.route('/', methods=['GET'])
def get_message():
    return jsonify(message='Hey, your app is working')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

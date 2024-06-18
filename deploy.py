from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

warnings.filterwarnings("ignore")

# Load the trained model (adjust the path as needed)
model = tf.keras.models.load_model('Model.h5', compile=False)
pred_395 = model.predict(np.array([[395]]))

data = pd.read_csv("https://raw.githubusercontent.com/MosesSinanta/Repository_1/main/dataset_1.csv")
data["Date / Time"] = pd.to_datetime(data["Date / Time"], format = "%d-%m-%Y")
data["Day"] = data["Date / Time"].dt.day
data["Month"] = data["Date / Time"].dt.month

days_in_month = {
    1: 31,
    2: 28,    # Considering non-leap year
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

data["Numerical Date"] = 0
data = data.drop_duplicates(subset = ["Numerical Date"], keep = "last")

data = data.drop(columns = ["Date / Time", "Mode", "Category", "Sub category", "Income / Expense", "Debit / Credit", "Day"])

X = data[["Numerical Date"]]
y = data["Cumulative"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

def calculate_growth(data):
    if data.empty:
        return 0
    start_value = data.iloc[0]["Cumulative"]
    end_value = data.iloc[-1]["Cumulative"]
    growth = ((end_value - start_value) / start_value) * 100
    return growth

oct_growth = calculate_growth(data[data["Month"] == 10])
nov_growth = calculate_growth(data[data["Month"] == 11])
dec_growth = calculate_growth(data[data["Month"] == 12])
avg_growth = np.mean([oct_growth, nov_growth, dec_growth])

threshold = 2.50    # Change according to group decision

start_value = data.iloc[-1]["Cumulative"]
end_value = pred_395
pred_growth = np.float64(((end_value - start_value) / start_value) * 100)

if (pred_growth >= avg_growth + threshold):
    print(f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangatlah baik!")
elif (pred_growth < avg_growth - threshold):
    print(f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% cukup buruk ;-;")
else:
    print(f"Pertumbuhan keuangan sebesar {pred_growth:.2f}% sangat stabil!")
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assuming the input is in JSON format and contains a key 'input_data'
    input_data = np.array(data['input_data']).reshape(1, -1)  # Adjust the reshape according to your model's input shape
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Assuming the model's prediction is a single value or a 1D array
    return jsonify({'prediction': prediction.tolist()})

@app.route('/', methods=['GET'])
def get_message():
    return jsonify(message='Hey, your app is working')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

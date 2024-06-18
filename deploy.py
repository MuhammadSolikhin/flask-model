from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

# Load the trained model (adjust the path as needed)
model = tf.keras.models.load_model('Model.h5', compile=False)

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

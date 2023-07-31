import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the model using joblib (if you exported using joblib)
loaded_model = joblib.load('linear_regression_model.joblib')

# OR, load the model using pickle (if you exported using pickle)
# import pickle
# with open('linear_regression_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return "hello farhan here"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_df = pd.DataFrame(data, index=[0])
    user_predictions = loaded_model.predict(user_df)
    return jsonify({'predicted_price': user_predictions[0]})

# This block should be at the end of your script
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port, e.g., 5001
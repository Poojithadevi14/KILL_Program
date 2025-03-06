from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # This loads the frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the frontend
        data = request.json  # JSON data from frontend
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Yes ( More likely to leave )" if prediction == 1 else "No ( Less likely to leave)"

        return jsonify({"prediction": result})  # Send response back to frontend
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

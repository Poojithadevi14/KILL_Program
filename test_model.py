import joblib
import numpy as np

# Load model
model = joblib.load("random_forest_model.pkl")

# Define medium-risk test input
example_input = np.array([[680, 1, 0, 38, 6, 75000, 2, 1, 0, 60000]])  

# Ensure the input shape matches the model
prediction = model.predict(example_input)

# Print the result
if prediction[0] == 1:
    print("Predicted Churn: Medium to High Chance (Customer may leave)")
else:
    print("Predicted Churn: Low to Medium Chance (Customer likely to stay)")

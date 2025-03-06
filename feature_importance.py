import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Get feature importances
importances = model.feature_importances_

# Define feature names (in the same order as training data)
feature_names = [
    "Credit Score", "Geography", "Gender", "Age", "Tenure", 
    "Balance", "Number of Products", "Has Credit Card", 
    "Is Active Member", "Estimated Salary"
]

print("Feature Importances:", importances)  # Debugging print

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Churn Prediction Model")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()

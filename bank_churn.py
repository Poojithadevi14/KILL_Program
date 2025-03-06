# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Step 1: Load Dataset
df = pd.read_csv("C:/Users/user/Desktop/team 7/train.csv")  # Update file path if needed
print("Dataset Loaded Successfully!\n", df.head())

# Step 2: Data Preprocessing
df.drop(columns=['id', 'CustomerId', 'Surname'], errors='ignore', inplace=True)  # Drop unnecessary columns

# Encoding categorical variables
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])  # Convert Geography to numeric (0,1,2)
df['Gender'] = le.fit_transform(df['Gender'])  # Convert Gender to numeric (0,1)

# Display processed dataset
print("Preprocessed Data:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 3: Split Data into Training & Testing Sets
X = df.drop(columns=['Churn'])  # Features
y = df['Churn']  # Target variable (1 = Churn, 0 = Not Churn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add this line to check the feature count
print("Number of features in training:", X_train.shape[1])  

# Step 4: Normalize Data (Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 6: Model Evaluation
print("\nLogistic Regression Performance:\n", classification_report(y_test, y_pred_log))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

print("\nRandom Forest Performance:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest AUC-ROC Score:", roc_auc_score(y_test, y_pred_rf))

# Step 7: Feature Importance (for Random Forest)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,5))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features in Churn Prediction")
plt.show()

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

print("✅ Model saved successfully as 'random_forest_model.pkl'")

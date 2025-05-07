import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

# Define file paths
DATA_DIR = "../data"
MODEL_DIR = "../model"
# Use the new dataset file
DATA_FILE = os.path.join(DATA_DIR, "lung_cancer_dataset_2_nancy.csv") 
# Save the new model with a distinct name
MODEL_FILE = os.path.join(MODEL_DIR, "lung_cancer_model_nancy.pkl") 
LE_TARGET_FILE = os.path.join(MODEL_DIR, "le_target_nancy.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

print("\n--- Initial Data Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

# --- Data Preprocessing --- 
print("\n--- Starting Data Preprocessing for Nancy Dataset ---")

# 1. Drop irrelevant columns
print("Dropping 'index' and 'Patient Id' columns...")
df = df.drop(["index", "Patient Id"], axis=1)

# 2. Clean column names (replace spaces, make lowercase, fix typos if any)
print("Cleaning column names...")
df.columns = df.columns.str.replace(" ", "_").str.replace("OccuPational", "Occupational").str.lower()
print("Cleaned columns:", df.columns.tolist())

# 3. Encode Target Variable ('level')
print("Encoding target variable 'level' (Low=0, Medium=1, High=2)...")
le_target = LabelEncoder()
# Ensure consistent order: Low, Medium, High
df["level"] = le_target.fit_transform(df["level"])
print("Target Level mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# 4. Handle Features
# Most features are ordinal (1-N). We'll treat them as numerical for Random Forest.
# Gender needs encoding (1=Male, 2=Female -> 0=Male, 1=Female or vice versa - let's check convention)
# Assuming 1=Male, 2=Female based on previous dataset. Let's map 1->0, 2->1
print("Encoding gender (1->0, 2->1)...")
if set(df["gender"].unique()) == {1, 2}:
    df["gender"] = df["gender"].replace({1: 0, 2: 1})
    print("Gender mapped to 0/1.")
else:
    print(f"Warning: Gender column has unexpected values: {df['gender'].unique()}")

print("\n--- Data After Preprocessing (First 5 Rows) ---")
print(df.head())

print("\n--- Final Value Counts for Target ---")
print(df["level"].value_counts())

# 5. Split Data
print("\n--- Splitting Data into Training and Testing Sets ---")
X = df.drop("level", axis=1)
y = df["level"]

# Get feature names after preprocessing for consistency
feature_names = X.columns.tolist()
print(f"Features for model: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- Model Training --- 
print("\n--- Starting Model Training (Random Forest Classifier) ---")
# Using RandomForestClassifier for multi-class classification
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=10, min_samples_split=5)
rf_model.fit(X_train, y_train)
print("Model training completed.")

# --- Model Evaluation --- 
print("\n--- Evaluating Model --- ")
y_pred = rf_model.predict(X_test)
# For multi-class, predict_proba gives probabilities for each class [Low, Medium, High]
y_pred_proba = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
# Use the fitted encoder's classes for target names
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save Model --- 
print(f"\n--- Saving Model to {MODEL_FILE} ---")
joblib.dump(rf_model, MODEL_FILE)
# Save the target encoder as well, as it maps Low/Medium/High to 0/1/2
joblib.dump(le_target, LE_TARGET_FILE)
# Save feature names list
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names_nancy.pkl"))
print("Model, target encoder, and feature names saved successfully.")

print("\n--- Feature Importances ---")
importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importances)

print("\n--- Script Finished ---")


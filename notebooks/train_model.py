import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define file paths
DATA_DIR = "../data"
MODEL_DIR = "../model"
DATA_FILE = os.path.join(DATA_DIR, "lung_cancer_dataset_1_mysar.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "lung_cancer_model.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

print("\n--- Initial Data Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Value Counts for Binary-like Columns ---")
for col in df.columns:
    if col not in ["AGE"]:
        print(f"\nValue counts for column: {col}")
        print(df[col].value_counts())

# --- Data Preprocessing --- 
print("\n--- Starting Data Preprocessing ---")

# 1. Clean column names (replace spaces and make lowercase)
print("Cleaning column names...")
df.columns = df.columns.str.replace(" ", "_").str.lower()
print("Cleaned columns:", df.columns.tolist())

# 2. Encode Categorical Features
print("Encoding categorical features (gender, lung_cancer)...")
# Gender: M=1, F=0 (as per dataset convention, though we'll use LabelEncoder)
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
print("Gender mapping:", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))

# Target: LUNG_CANCER: YES=1, NO=0
le_cancer = LabelEncoder()
df["lung_cancer"] = le_cancer.fit_transform(df["lung_cancer"])
print("Lung Cancer mapping:", dict(zip(le_cancer.classes_, le_cancer.transform(le_cancer.classes_))))

# 3. Convert other binary features (currently 1/2) to 0/1
# Assuming 1 -> 0 (No) and 2 -> 1 (Yes) based on common conventions
print("Converting binary features (1/2) to (0/1)...")
binary_cols = [
    'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
    'fatigue_', 'allergy_', 'wheezing', 'alcohol_consuming', 'coughing',
    'shortness_of_breath', 'swallowing_difficulty', 'chest_pain'
]

for col in binary_cols:
    if col in df.columns:
        # Check if values are indeed 1 and 2
        unique_vals = df[col].unique()
        if set(unique_vals) == {1, 2}:
            df[col] = df[col].replace({1: 0, 2: 1})
            print(f"Converted {col} to 0/1.")
        else:
            print(f"Warning: Column {col} does not contain only values 1 and 2. Unique values: {unique_vals}. Skipping conversion.")
    else:
        print(f"Warning: Expected binary column {col} not found in dataframe.")

print("\n--- Data After Preprocessing (First 5 Rows) ---")
print(df.head())

print("\n--- Final Value Counts ---")
for col in df.columns:
    if col not in ["age"]:
        print(f"\nValue counts for column: {col}")
        print(df[col].value_counts())

# 4. Split Data
print("\n--- Splitting Data into Training and Testing Sets ---")
X = df.drop("lung_cancer", axis=1)
y = df["lung_cancer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- Model Training --- 
print("\n--- Starting Model Training (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
print("Model training completed.")

# --- Model Evaluation --- 
print("\n--- Evaluating Model --- ")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1] # Probability of class 1 (YES)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_cancer.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save Model --- 
print(f"\n--- Saving Model to {MODEL_FILE} ---")
joblib.dump(rf_model, MODEL_FILE)
# Also save the label encoders if needed for interpretation later
joblib.dump(le_gender, os.path.join(MODEL_DIR, "le_gender.pkl"))
joblib.dump(le_cancer, os.path.join(MODEL_DIR, "le_cancer.pkl"))
print("Model and encoders saved successfully.")

print("\n--- Feature Importances ---")
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

print("\n--- Script Finished ---")


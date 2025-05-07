import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os
import numpy as np

# Define file paths
DATA_DIR = "../data"
MODEL_DIR = "../model"
DATA_FILE = os.path.join(DATA_DIR, "lung_cancer_dataset_2_nancy.csv") 

# Model and component file names
RF_MODEL_FILE = os.path.join(MODEL_DIR, "rf_model_normalized.pkl") 
XGB_MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model_normalized.pkl")
LR_MODEL_FILE = os.path.join(MODEL_DIR, "lr_model_normalized.pkl")
LE_TARGET_FILE = os.path.join(MODEL_DIR, "le_target_normalized.pkl")
SCALER_AGE_FILE = os.path.join(MODEL_DIR, "scaler_age_normalized.pkl")
ORDINAL_MAP_FILE = os.path.join(MODEL_DIR, "ordinal_map_normalized.pkl")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names_normalized.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

print("\n--- Initial Data Info ---")
df.info()

# --- Data Preprocessing --- 
print("\n--- Starting Data Preprocessing & Normalization ---")

# 1. Drop irrelevant columns
print("Dropping 'index' and 'Patient Id' columns...")
df = df.drop(["index", "Patient Id"], axis=1)

# 2. Clean column names
print("Cleaning column names...")
df.columns = df.columns.str.replace(" ", "_").str.replace("OccuPational", "Occupational").str.lower()
print("Cleaned columns:", df.columns.tolist())

# 3. Encode Target Variable ('level')
print("Encoding target variable 'level'...")
le_target = LabelEncoder()
df["level"] = le_target.fit_transform(df["level"])
class_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"Target Level mapping: {class_mapping}") # e.g., {'High': 0, 'Low': 1, 'Medium': 2}

# 4. Encode Gender
print("Encoding gender (1->0, 2->1)...")
if set(df["gender"].unique()) == {1, 2}:
    df["gender"] = df["gender"].replace({1: 0, 2: 1})
    print("Gender mapped to 0/1.")
else:
    print(f"Warning: Gender column has unexpected values: {df['gender'].unique()}")

# 5. Normalize Ordinal Features to 0-10 and Scale Age to 0-1
print("Normalizing ordinal features to 0-10 and scaling age to 0-1...")
ordinal_cols_ranges = {
    'air_pollution': (1, 8),
    'alcohol_use': (1, 8),
    'dust_allergy': (1, 8),
    'occupational_hazards': (1, 8),
    'genetic_risk': (1, 7),
    'chronic_lung_disease': (1, 7),
    'balanced_diet': (1, 7),
    'obesity': (1, 7),
    'smoking': (1, 8),
    'passive_smoker': (1, 8),
    'chest_pain': (1, 9),
    'coughing_of_blood': (1, 9),
    'fatigue': (1, 9),
    'weight_loss': (1, 8),
    'shortness_of_breath': (1, 9),
    'wheezing': (1, 8),
    'swallowing_difficulty': (1, 8),
    'clubbing_of_finger_nails': (1, 9),
    'frequent_cold': (1, 7),
    'dry_cough': (1, 7),
    'snoring': (1, 7)
}
ordinal_map_details = {}
for col, (min_orig, max_orig) in ordinal_cols_ranges.items():
    if col in df.columns:
        df[col] = ((df[col] - min_orig) / (max_orig - min_orig)) * 10
        ordinal_map_details[col] = {'min_orig': min_orig, 'max_orig': max_orig}
    else:
        print(f"Warning: Column {col} not found for normalization.")

scaler_age = MinMaxScaler()
df['age'] = scaler_age.fit_transform(df[['age']])
print("Age scaled to [0, 1] using MinMaxScaler.")

print("\n--- Data After Normalization/Scaling (First 5 Rows) ---")
print(df.head())

# 6. Split Data
print("\n--- Splitting Data into Training and Testing Sets ---")
X = df.drop("level", axis=1)
y = df["level"]
feature_names = X.columns.tolist()
print(f"Features for model: {feature_names}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- Model Training --- 

# Random Forest
print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=10, min_samples_split=5)
rf_model.fit(X_train, y_train)
print("RF Model training completed.")

# XGBoost
print("\n--- Training XGBoost Classifier ---")
# Note: XGBoost requires target labels to be 0 to num_class-1. Our LabelEncoder does this.
xgb_model = XGBClassifier(objective='multi:softprob', num_class=len(le_target.classes_), 
                          eval_metric='mlogloss', use_label_encoder=False, 
                          random_state=42, n_estimators=100)
xgb_model.fit(X_train, y_train)
print("XGBoost Model training completed.")

# Logistic Regression
print("\n--- Training Logistic Regression Classifier ---")
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, multi_class='ovr')
lr_model.fit(X_train, y_train)
print("Logistic Regression Model training completed.")

# --- Model Evaluation --- 

print("\n--- Evaluating Models --- ")
models = {"Random Forest": rf_model, "XGBoost": xgb_model, "Logistic Regression": lr_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- Save Models and Components --- 
print(f"\n--- Saving Models and Components ---")
joblib.dump(rf_model, RF_MODEL_FILE)
joblib.dump(xgb_model, XGB_MODEL_FILE)
joblib.dump(lr_model, LR_MODEL_FILE)
joblib.dump(le_target, LE_TARGET_FILE)
joblib.dump(scaler_age, SCALER_AGE_FILE)
joblib.dump(ordinal_map_details, ORDINAL_MAP_FILE)
joblib.dump(feature_names, FEATURE_NAMES_FILE)
print("All models, target encoder, age scaler, ordinal map, and feature names saved successfully.")

# --- Feature Importances (if available) --- 
print("\n--- Feature Importances ---")
if hasattr(rf_model, 'feature_importances_'):
    print("\nRandom Forest:")
    importances_rf = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(importances_rf.head(10))
if hasattr(xgb_model, 'feature_importances_'):
    print("\nXGBoost:")
    importances_xgb = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(importances_xgb.head(10))
# Logistic Regression coefficients can also be inspected but represent importance differently

print("\n--- Script Finished ---")


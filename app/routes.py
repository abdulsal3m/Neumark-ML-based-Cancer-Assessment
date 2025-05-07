from flask import render_template, request, jsonify, url_for, redirect, session # Added session
from app import app
import joblib
import pandas as pd
import os
import numpy as np

# Define model paths for the normalized models
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
RF_MODEL_FILE = os.path.join(MODEL_DIR, "rf_model_normalized.pkl") 
XGB_MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model_normalized.pkl")
LR_MODEL_FILE = os.path.join(MODEL_DIR, "lr_model_normalized.pkl")
LE_TARGET_FILE = os.path.join(MODEL_DIR, "le_target_normalized.pkl")
SCALER_AGE_FILE = os.path.join(MODEL_DIR, "scaler_age_normalized.pkl")
ORDINAL_MAP_FILE = os.path.join(MODEL_DIR, "ordinal_map_normalized.pkl") # Needed for info, not transform
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names_normalized.pkl")

# Load models and components
print("Loading models and components...")
try:
    model_rf = joblib.load(RF_MODEL_FILE)
    model_xgb = joblib.load(XGB_MODEL_FILE)
    model_lr = joblib.load(LR_MODEL_FILE)
    le_target = joblib.load(LE_TARGET_FILE)
    scaler_age = joblib.load(SCALER_AGE_FILE)
    ordinal_map_details = joblib.load(ORDINAL_MAP_FILE) # Load for reference if needed
    expected_features = joblib.load(FEATURE_NAMES_FILE)
    print("Models and components loaded successfully.")
    print(f"Expected features: {expected_features}")
    print(f"Target classes: {le_target.classes_.tolist()}") # e.g., [High, Low, Medium]
    
    # Identify the index for the 'High' class
    high_class_index = np.where(le_target.classes_ == 'High')[0][0]
    print(f"Index for 'High' class: {high_class_index}")

    models = {"Random Forest": model_rf, "XGBoost": model_xgb, "Logistic Regression": model_lr}

except FileNotFoundError as e:
    print(f"Error loading model file: {e}")
    models = {}
    le_target = None
    scaler_age = None
    expected_features = []
    high_class_index = -1 # Indicate error
except Exception as e:
    print(f"Error loading model components: {e}")
    models = {}
    le_target = None
    scaler_age = None
    expected_features = []
    high_class_index = -1

# Define factor categorization logic (Updated)
# Thresholds: low <= 3, high >= 7. Values 4, 5, 6 are neutral.
# Type: 'risk_high' (high value is risk), 'risk_low' (low value is risk)
FACTOR_CATEGORIZATION_CONFIG = {
    'Air Pollution': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Alcohol use': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Dust Allergy': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Occupational Hazards': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Genetic Risk': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Chronic Lung Disease': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Balanced Diet': {'type': 'risk_low', 'low_threshold': 4, 'high_threshold': 6}, # Low diet score is risk
    'Obesity': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Smoking': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Passive Smoker': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Chest Pain': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Coughing of Blood': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Fatigue': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Weight Loss': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Shortness of Breath': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Wheezing': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Swallowing Difficulty': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Clubbing of Finger Nails': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Frequent Cold': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Dry Cough': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Snoring': {'type': 'risk_high', 'low_threshold': 4, 'high_threshold': 6},
    'Age': {'type': 'neutral'},
    'Gender': {'type': 'neutral'}
}

# Specific actionable recommendations for each factor
FACTOR_RECOMMENDATIONS = {
    'Smoking': "Quit smoking and avoid tobacco products completely.",
    'Passive Smoker': "Avoid secondhand smoke exposure in all environments.",
    'Air Pollution': "Use air purifiers at home and wear masks in polluted areas.",
    'Alcohol use': "Limit alcohol consumption to occasional use only.",
    'Dust Allergy': "Install HEPA filters and maintain regular cleaning routines.",
    'Occupational Hazards': "Use proper PPE and request workplace safety assessments.",
    'Genetic Risk': "Schedule regular screenings with your healthcare provider.",
    'Chronic Lung Disease': "Follow your treatment plan and avoid respiratory irritants.",
    'Balanced Diet': "Increase intake of fruits, vegetables, and whole grains.",
    'Obesity': "Work with a nutritionist to develop a sustainable weight management plan.",
    'Chest Pain': "Seek immediate medical evaluation for any chest discomfort.",
    'Coughing of Blood': "This requires urgent medical attention - consult a doctor immediately.",
    'Fatigue': "Evaluate sleep quality and consider a medical check-up.",
    'Weight Loss': "Unexplained weight loss should be evaluated by a healthcare provider.",
    'Shortness of Breath': "Consult a pulmonologist for proper evaluation and treatment.",
    'Wheezing': "Avoid triggers and use prescribed inhalers as directed.",
    'Swallowing Difficulty': "See a specialist for evaluation of potential esophageal issues.",
    'Clubbing of Finger Nails': "This can indicate lung disease - consult a doctor promptly.",
    'Frequent Cold': "Boost immunity with proper nutrition and consider allergy testing.",
    'Dry Cough': "Stay hydrated and identify potential environmental triggers.",
    'Snoring': "Consider sleep evaluation for potential sleep apnea."
}

@app.route("/")
def index():
    """Landing page route."""
    return render_template("index.html", title="Neumark Home")

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    """Questionnaire page route."""
    if request.method == "POST":
        if not models or not le_target or not scaler_age or not expected_features or high_class_index == -1:
            return "Error: Model components not available. Please check server logs.", 500
            
        try:
            user_data_raw = request.form.to_dict()
            user_input_data_for_session = {}
            for key, value in user_data_raw.items():
                try:
                    num_value = float(value)
                    user_input_data_for_session[key] = int(num_value) if num_value.is_integer() else num_value
                except ValueError:
                    user_input_data_for_session[key] = value
            session["user_input_data"] = user_input_data_for_session

            processed_data = {}
            feature_map_form_to_model = {
                'Age': 'age', 'Gender': 'gender', 'Air Pollution': 'air_pollution',
                'Alcohol use': 'alcohol_use', 'Dust Allergy': 'dust_allergy',
                'Occupational Hazards': 'occupational_hazards', 'Genetic Risk': 'genetic_risk',
                'Chronic Lung Disease': 'chronic_lung_disease', 'Balanced Diet': 'balanced_diet',
                'Obesity': 'obesity', 'Smoking': 'smoking', 'Passive Smoker': 'passive_smoker',
                'Chest Pain': 'chest_pain', 'Coughing of Blood': 'coughing_of_blood',
                'Fatigue': 'fatigue', 'Weight Loss': 'weight_loss',
                'Shortness of Breath': 'shortness_of_breath', 'Wheezing': 'wheezing',
                'Swallowing Difficulty': 'swallowing_difficulty',
                'Clubbing of Finger Nails': 'clubbing_of_finger_nails',
                'Frequent Cold': 'frequent_cold', 'Dry Cough': 'dry_cough', 'Snoring': 'snoring'
            }
            for form_key, model_key in feature_map_form_to_model.items():
                form_value = user_data_raw.get(form_key)
                if form_value is None:
                    processed_data[model_key] = 0
                    continue
                try:
                    if model_key == 'age':
                        processed_data[model_key] = float(scaler_age.transform(np.array([[int(form_value)]]))[0, 0])
                    elif model_key == 'gender':
                        processed_data[model_key] = 1 if form_value == '2' else 0
                    else:
                        processed_data[model_key] = int(form_value)
                except ValueError:
                     processed_data[model_key] = 0

            input_df = pd.DataFrame([processed_data], columns=expected_features)

            results = {}
            high_risk_probs = [] # Stores raw probabilities for RF, XGB, LR in order
            # Model accuracies to be used as weights (RF, XGB, LR)
            model_weights = np.array([0.9000, 0.8950, 0.8050])

            # Ensure the order of processing matches the weights
            # The models dictionary is {"Random Forest": model_rf, "XGBoost": model_xgb, "Logistic Regression": model_lr}
            # So, iterating through models.items() will give RF, then XGB, then LR if Python version >= 3.7 (dict order preserved)
            # For safety, explicitly process in the order of weights if needed, or rely on dict order for Python 3.7+
            # Current loop order (RF, XGB, LR) matches the intended weights order.
            for name, model in models.items():
                pred_proba = model.predict_proba(input_df)[0]
                high_prob = pred_proba[high_class_index]
                high_risk_probs.append(high_prob) # Appends RF_prob, then XGB_prob, then LR_prob
                results[name] = {
                    'high_prob': round(float(high_prob) * 100, 1),
                    'predicted_level': le_target.classes_[np.argmax(pred_proba)]
                }

            # Calculate WEIGHTED average probability and overall level
            if len(high_risk_probs) == len(model_weights):
                weighted_average_high_prob = np.average(high_risk_probs, weights=model_weights)
            else:
                print("Error: Mismatch in number of probabilities and weights. Using simple average.")
                weighted_average_high_prob = np.mean(high_risk_probs) # Fallback
            
            weighted_average_high_prob_percent = round(float(weighted_average_high_prob) * 100, 1)
            print(f"Weighted Average High Risk Probability: {weighted_average_high_prob:.4f} ({weighted_average_high_prob_percent}%)")
            
            if weighted_average_high_prob > 0.6: overall_level = 'High'
            elif weighted_average_high_prob > 0.3: overall_level = 'Medium'
            else: overall_level = 'Low'
            print(f"Overall Determined Level (Weighted): {overall_level}")

            categorized_factors = {'positive': {}, 'risk': {}}
            for factor_name, value in user_input_data_for_session.items():
                if factor_name in FACTOR_CATEGORIZATION_CONFIG:
                    config = FACTOR_CATEGORIZATION_CONFIG[factor_name]
                    if config['type'] == 'risk_high':
                        if value >= config['high_threshold']:
                            categorized_factors['risk'][factor_name] = value
                        elif value <= config['low_threshold']:
                            categorized_factors['positive'][f"Low {factor_name}"] = 10 - value
                    elif config['type'] == 'risk_low':
                        if value <= config['low_threshold']:
                            categorized_factors['risk'][factor_name] = 10 - value # Higher value for risk pie chart
                        elif value >= config['high_threshold']:
                            categorized_factors['positive'][f"High {factor_name}"] = value
            session["categorized_factors"] = categorized_factors

            recommendations = []
            risk_factors_sorted = sorted(categorized_factors['risk'].items(), key=lambda x: x[1], reverse=True)
            for factor, value in risk_factors_sorted[:3]:
                clean_factor_name = factor.replace("Low ", "") # Handle cases like "Low Balanced Diet"
                if factor in FACTOR_RECOMMENDATIONS:
                    recommendations.append(FACTOR_RECOMMENDATIONS[factor])
                elif clean_factor_name in FACTOR_RECOMMENDATIONS:
                     recommendations.append(FACTOR_RECOMMENDATIONS[clean_factor_name])
            
            if overall_level == 'High':
                recommendations.append("Schedule a lung cancer screening with your doctor.")
            elif overall_level == 'Medium':
                recommendations.append("Discuss these risk factors with your healthcare provider.")
            else:
                if not recommendations: recommendations.append("Continue your healthy lifestyle choices.")
            
            session["results_data"] = {
                "overall_level": overall_level,
                "avg_score": float(weighted_average_high_prob_percent), # Use the weighted score
                "rf_score": float(results['Random Forest']['high_prob']),
                "xgb_score": float(results['XGBoost']['high_prob']),
                "lr_score": float(results['Logistic Regression']['high_prob']),
                "recommendations": recommendations
            }
            
            return redirect(url_for("results"))

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"An error occurred during processing: {e}", 500

    return render_template("questionnaire.html", title="Start Assessment")

@app.route("/results")
def results():
    results_data = session.get("results_data", None)
    user_input_data = session.get("user_input_data", None)
    categorized_factors = session.get("categorized_factors", None)
    if not results_data or not user_input_data or not categorized_factors:
        return redirect(url_for("assessment"))
    return render_template("results.html", title="Assessment Results", results=results_data, user_data=user_input_data, categorized_factors=categorized_factors)

@app.route("/about-us")
def about_us():
    return render_template("about_us.html", title="About Neumark")


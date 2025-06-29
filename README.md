# Neumark: Lung Cancer Risk Assessment Tool

**A real-time, browser-based risk assessment application using ensemble ML models for early lung cancer detection.**

---

## Overview

Neumark was built to bridge the accessibility gap in preventative healthcare. This full-stack web application delivers personalized lung cancer risk predictions based on lifestyle, symptoms, and exposure history—entirely online and in under 3 minutes.

Powered by a trained ensemble of machine learning models and deployed via Flask and Render, Neumark emphasizes explainability, speed, and user experience, making predictive health analytics accessible to everyone.

---

## Features

- **Ensemble Machine Learning**  
  Combines Random Forest, Logistic Regression, and XGBoost to generate a robust composite risk score with interpretable outputs.

- **Intuitive, Accessible UI**  
  One-question-per-screen format with real-time validation and responsive design.

- **Fast & Confidential Inference**  
  Sub-second predictions with zero user data stored.

- **Zero-install Deployment**  
  Hosted via Render.com using Flask + Gunicorn backend and GitHub CI/CD.

---

## Tech Stack

| Layer             | Tools/Frameworks                                                   |
|------------------|---------------------------------------------------------------------|
| Backend           | Flask, Gunicorn, Jinja2, Python 3                                  |
| Frontend          | HTML, CSS, JavaScript (vanilla)                                    |
| Machine Learning  | scikit-learn, XGBoost, Logistic Regression, Random Forest, Pandas  |
| DevOps/Hosting    | Render.com, GitHub Actions (CI/CD), joblib (model serialization)   |

---

## Model Performance

| Model              | Accuracy | Macro F1 | Notes                                       |
|-------------------|----------|----------|---------------------------------------------|
| Random Forest      | 90.0%    | 0.90     | Strong generalization, high recall          |
| XGBoost            | 89.5%    | 0.89     | Balanced output across risk tiers           |
| Logistic Regression| 80.5%    | 0.78     | Strong for high-risk predictions            |

> Final ensemble weights optimized for AUC and interpretability.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/neumark-lung-cancer-risk

# Install dependencies
pip install -r requirements.txt

# Run the app
python run.py

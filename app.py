"""
app.py
Flask backend for the Human Development Index (HDI) Predictor web app.

Routes:
  GET  /            -> Home / prediction input form (indexnew.html)
  POST /predict      -> Handle form submission, run prediction, render result

The trained Linear Regression model is loaded once at startup from
model/HDI.pkl (pickle serialization) and reused for every request.
"""

import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "HDI.pkl")

# ---------------------------------------------------------------------------
# Load the trained model once at startup (Runtime Model Loading)
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_ORDER = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
]

# A small illustrative country dropdown list for the frontend (optional demo data)
SAMPLE_COUNTRIES = [
    "Norway", "Switzerland", "Germany", "United States", "United Kingdom",
    "Japan", "South Korea", "China", "Brazil", "India", "South Africa",
    "Nigeria", "Bangladesh", "Ethiopia", "Custom / Other",
]


def classify_hdi(score: float) -> str:
    """Map a predicted HDI score to its official UNDP category."""
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


@app.route("/")
def home():
    """Render the home page with the prediction input form."""
    return render_template("indexnew.html", countries=SAMPLE_COUNTRIES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle POST request:
      1. Validate user input
      2. Convert inputs into a feature vector
      3. Run the Linear Regression model
      4. Map predicted score to an HDI category
      5. Render the result page
    """
    try:
        country = request.form.get("country", "N/A")
        life_expectancy = float(request.form["life_expectancy"])
        mean_schooling = float(request.form["mean_schooling"])
        expected_schooling = float(request.form["expected_schooling"])
        gni_per_capita = float(request.form["gni_per_capita"])

        # ---- basic input validation ----
        errors = []
        if not (0 < life_expectancy <= 100):
            errors.append("Life expectancy must be between 0 and 100 years.")
        if not (0 <= mean_schooling <= 20):
            errors.append("Mean years of schooling must be between 0 and 20.")
        if not (0 <= expected_schooling <= 25):
            errors.append("Expected years of schooling must be between 0 and 25.")
        if gni_per_capita <= 0:
            errors.append("GNI per capita must be a positive number.")

        if errors:
            return render_template(
                "indexnew.html", countries=SAMPLE_COUNTRIES, errors=errors
            )

        # ---- feature vector + prediction ----
        feature_vector = np.array(
            [[life_expectancy, mean_schooling, expected_schooling, gni_per_capita]]
        )
        predicted_score = float(model.predict(feature_vector)[0])
        predicted_score = max(0.0, min(1.0, predicted_score))  # clamp to [0,1]
        category = classify_hdi(predicted_score)

        return render_template(
            "resultnew.html",
            country=country,
            life_expectancy=life_expectancy,
            mean_schooling=mean_schooling,
            expected_schooling=expected_schooling,
            gni_per_capita=gni_per_capita,
            hdi_score=round(predicted_score, 4),
            hdi_category=category,
        )

    except (KeyError, ValueError):
        return render_template(
            "indexnew.html",
            countries=SAMPLE_COUNTRIES,
            errors=["Please fill in all fields with valid numeric values."],
        )


if __name__ == "__main__":
    app.run(debug=True)

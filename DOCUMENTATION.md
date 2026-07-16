# 📘 HDI Predictor — Project Documentation

A complete technical reference for the **Human Development Index (HDI) Predictor**: architecture, data pipeline, model details, API reference, and setup instructions.

---

## 1. Overview

The HDI Predictor is an end-to-end Machine Learning web application that estimates a country's **Human Development Index (HDI)** score from four socio-economic indicators, and classifies it into an official UNDP development category.

| | |
|---|---|
| **Problem type** | Regression (continuous HDI score, 0–1) |
| **Model** | Linear Regression (scikit-learn) |
| **Interface** | Flask web app (HTML/CSS forms) |
| **Input features** | Life Expectancy, Mean Years of Schooling, Expected Years of Schooling, GNI per Capita |
| **Output** | Predicted HDI score + category (Low / Medium / High / Very High) |

---

## 2. System Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  User Layer │ ───▶ │  Flask App Layer  │ ───▶ │  Prediction Engine │
│ (Browser)   │ ◀─── │  app.py           │ ◀─── │  model/HDI.pkl     │
└─────────────┘      └──────────────────┘      └───────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Templates /CSS   │
                     │  indexnew.html    │
                     │  resultnew.html   │
                     └──────────────────┘
```

**Request flow:**
1. User opens `/` → sees the prediction form (`indexnew.html`).
2. User submits the form → `POST /predict`.
3. `app.py` validates input, builds a 4-feature vector, and calls `model.predict()`.
4. The predicted score (clamped to `[0, 1]`) is mapped to a category.
5. `resultnew.html` renders the score, category, and input summary.

---

## 3. Repository Structure

```
hdi-predictor/
├── app.py                     # Flask backend: routes, validation, prediction
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # Quick-start guide
├── DOCUMENTATION.md            # This file
├── .gitignore
│
├── data/
│   ├── generate_dataset.py     # Builds the HDI dataset from the UNDP formula
│   └── hdi_dataset.csv         # 159-country dataset used for training
│
├── model/
│   ├── train_model.py          # EDA → preprocessing → training → evaluation
│   ├── HDI.pkl                 # Trained Linear Regression model (pickle)
│   ├── metrics.txt             # Saved R² / MAE / RMSE scores
│   └── eda_plots/              # strip_plot, distplots, heatmap, scatterplots,
│                                # actual_vs_predicted (all PNG)
│
├── templates/
│   ├── indexnew.html           # Input form page
│   └── resultnew.html          # Result display page
│
└── static/
    └── style.css                # App styling
```

---

## 4. Dataset

**File:** `data/hdi_dataset.csv` (159 rows, 7 columns)

| Column | Type | Description |
|---|---|---|
| `Country` | string | Country name |
| `Life_Expectancy` | float | Life expectancy at birth (years) |
| `Mean_Years_Schooling` | float | Average years of schooling completed (25+ population) |
| `Expected_Years_Schooling` | float | Expected years of schooling for a child entering school |
| `GNI_Per_Capita` | float | Gross National Income per capita (USD, PPP) |
| `HDI_Score` | float | Target variable — Human Development Index (0–1) |
| `HDI_Category` | string | Low / Medium / High / Very High |

**How it was generated** (`data/generate_dataset.py`):
The dataset is built programmatically using the **official UNDP HDI formula**, rather than downloaded, so the project is fully self-contained and reproducible:

```
Life Expectancy Index = (LE − 20) / (85 − 20)
Education Index       = (MYS/15 + EYS/18) / 2
Income Index          = (ln(GNI) − ln(100)) / (ln(75000) − ln(100))
HDI                   = (LE_Index × Edu_Index × Income_Index) ^ (1/3)
```

Countries are assigned a realistic development tier (matching real-world HDI rankings) before sampling feature values, so well-known countries land in believable brackets. A small amount of Gaussian noise is added to the final HDI score, and a handful of values are set to `NaN` to make the preprocessing step meaningful.

> To use **real UNDP/Kaggle data** instead, replace `hdi_dataset.csv` with a file using the same column names, then re-run `model/train_model.py`.

---

## 5. Machine Learning Pipeline (`model/train_model.py`)

| Step | Description |
|---|---|
| **1. Load data** | Reads `data/hdi_dataset.csv` with Pandas; reports missing values. |
| **2. EDA** | Generates strip plots, distribution plots, a correlation heatmap, and scatter plots (saved to `model/eda_plots/`). |
| **3. Preprocessing** | Mean imputation for null values; label encoding of `HDI_Category` for reference. |
| **4. Train/test split** | 75% train / 25% test (`random_state=42`). |
| **5. Model training** | `sklearn.linear_model.LinearRegression` fit on 4 features. |
| **6. Evaluation** | R² score, MAE, RMSE, and an Actual-vs-Predicted scatter plot. |
| **7. Serialization** | Model saved to `model/HDI.pkl` via `pickle.dump()`. |

**Latest results** (see `model/metrics.txt`):

| Metric | Value |
|---|---|
| Train R² | 0.988 |
| Test R² | 0.984 |
| Test MAE | 0.019 |
| Test RMSE | 0.027 |

---

## 6. Flask Application (`app.py`)

### Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the input form (`indexnew.html`) |
| `/predict` | POST | Validates form input, runs the model, renders `resultnew.html` |

### `/predict` — Request/Response

**Form fields expected:**
```
country              (string)
life_expectancy      (float, 0–100)
mean_schooling       (float, 0–20)
expected_schooling   (float, 0–25)
gni_per_capita       (float, > 0)
```

**Validation:** Each field is range-checked server-side; invalid or missing values redisplay the form with an error banner instead of crashing.

**Prediction logic:**
```python
feature_vector = [[life_expectancy, mean_schooling, expected_schooling, gni_per_capita]]
predicted_score = model.predict(feature_vector)[0]
predicted_score = clamp(predicted_score, 0, 1)
category = classify_hdi(predicted_score)
```

### HDI Category Thresholds

| Category | Score Range |
|---|---|
| Very High | ≥ 0.80 |
| High | 0.70 – 0.799 |
| Medium | 0.55 – 0.699 |
| Low | < 0.55 |

---

## 7. Setup & Usage

```bash
# 1. Clone
git clone https://github.com/<your-username>/hdi-predictor.git
cd hdi-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Regenerate dataset & retrain model
python data/generate_dataset.py
python model/train_model.py

# 5. Run the app
python app.py
```

Open **http://127.0.0.1:5000/** in your browser.

---

## 8. Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Web framework | Flask |
| ML | scikit-learn (LinearRegression) |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Pickle |
| Frontend | HTML, CSS (Jinja2 templating) |

---

## 9. Possible Extensions

- Swap `LinearRegression` for `RandomForestRegressor` or `XGBoost` and compare R².
- Add a `/api/predict` JSON endpoint for programmatic access.
- Deploy to Render, Railway, or Heroku with a `Procfile`.
- Add a Jupyter notebook (`notebooks/EDA.ipynb`) walking through the analysis interactively.
- Replace the synthetic dataset with the real UNDP Human Development Report data.

---

## 10. License

MIT License — free to use, modify, and distribute for learning and portfolio purposes.

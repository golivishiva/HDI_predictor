# 🌍 Human Development Index (HDI) Predictor

An end-to-end Machine Learning web application that predicts a country's
**Human Development Index (HDI)** score from four key indicators:

- Life Expectancy
- Mean Years of Schooling
- Expected Years of Schooling
- GNI (Gross National Income) Per Capita

Built with **Python, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.**

---

## 📁 Project Structure

```
hdi-predictor/
├── app.py                     # Flask backend (routes, prediction logic)
├── requirements.txt           # Python dependencies
├── data/
│   ├── generate_dataset.py    # Synthetic-but-realistic HDI dataset generator
│   └── hdi_dataset.csv        # Dataset (159 countries)
├── model/
│   ├── train_model.py         # EDA + preprocessing + training + evaluation
│   ├── HDI.pkl                # Trained Linear Regression model (pickle)
│   ├── metrics.txt            # Saved evaluation metrics
│   └── eda_plots/             # Strip plots, distplots, heatmap, scatterplots
├── templates/
│   ├── indexnew.html          # Prediction input form
│   └── resultnew.html         # Prediction result page
└── static/
    └── style.css               # App styling
```

---

## 🧠 How It Works (Pipeline)

1. **Environment Setup** – NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
2. **Dataset** – `data/hdi_dataset.csv` contains 159 country-level records with
   Life Expectancy, Schooling, GNI per Capita, HDI Score, and HDI Category,
   generated with the official UNDP HDI formula plus realistic noise.
3. **EDA** – Strip plots, distribution plots, correlation heatmap, and
   scatter plots (see `model/eda_plots/`).
4. **Preprocessing** – Mean imputation for missing values, label encoding
   for the HDI category, 75/25 train-test split.
5. **Model Training** – `LinearRegression` from scikit-learn.
6. **Evaluation** – R² score, MAE, RMSE, and an Actual vs Predicted plot
   (Test R² ≈ 0.98 on this dataset).
7. **Serialization** – Trained model saved to `model/HDI.pkl` with `pickle`.
8. **Flask Web App** – `app.py` loads `HDI.pkl` at startup and serves:
   - `GET /` → prediction input form
   - `POST /predict` → validates input, builds the feature vector, runs the
     model, maps the score to a category (Low / Medium / High / Very High),
     and renders the result page.

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/hdi-predictor.git
cd hdi-predictor
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset and retrain the model
```bash
python data/generate_dataset.py
python model/train_model.py
```
This regenerates `data/hdi_dataset.csv`, the EDA plots, and re-saves
`model/HDI.pkl`. A pre-trained model is already included, so this step is
optional.

### 3. Run the Flask app
```bash
python app.py
```
Then open **http://127.0.0.1:5000/** in your browser.

---

## 🎯 HDI Category Thresholds

| Category   | HDI Score Range |
|------------|------------------|
| Very High  | ≥ 0.80           |
| High       | 0.70 – 0.799     |
| Medium     | 0.55 – 0.699     |
| Low        | < 0.55           |

---

## 🛠️ Technology Stack

Python · Flask · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Jupyter

---

## 📌 Notes

- The dataset in this repo is **synthetically generated** using the official
  UNDP HDI methodology (with added noise) so the project is fully
  self-contained and reproducible without external downloads. To use real
  UNDP data instead, replace `data/hdi_dataset.csv` with a dataset from
  [Kaggle](https://www.kaggle.com/) or [hdr.undp.org](https://hdr.undp.org/)
  keeping the same column names, then re-run `model/train_model.py`.
- This is an educational project intended to demonstrate an end-to-end
  ML + Flask deployment workflow, not an official HDI data source.

## 📄 License

MIT License — free to use and modify for learning purposes.

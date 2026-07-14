"""
train_model.py
End-to-end ML pipeline for the HDI Predictor project.

Steps:
 1. Load dataset
 2. Exploratory Data Analysis (EDA) -> saves plots to model/eda_plots/
 3. Data preprocessing (null handling via mean imputation, label encoding)
 4. Train/test split (75/25)
 5. Train Linear Regression model
 6. Evaluate (R2 score, train/test accuracy, actual vs predicted plot)
 7. Serialize trained model with Pickle -> model/HDI.pkl
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "hdi_dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")
MODEL_PATH = os.path.join(BASE_DIR, "HDI.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ["Life_Expectancy", "Mean_Years_Schooling", "Expected_Years_Schooling", "GNI_Per_Capita"]
TARGET = "HDI_Score"

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("Step 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Missing values per column:\n{df.isnull().sum()}\n")

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
print("Step 2: Running EDA and saving plots...")

# Strip plots
plt.figure(figsize=(8, 5))
sns.stripplot(x="HDI_Category", y="HDI_Score", data=df,
              order=["Low", "Medium", "High", "Very High"])
plt.title("HDI Score Strip Plot by Category")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "strip_plot.png"))
plt.close()

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "distplots.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(7, 6))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"))
plt.close()

# Scatter plots vs target
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, FEATURES):
    sns.scatterplot(x=col, y=TARGET, data=df, ax=ax)
    ax.set_title(f"{col} vs HDI Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "scatterplots.png"))
plt.close()
print(f"  Plots saved to {PLOTS_DIR}\n")

# ---------------------------------------------------------------------------
# 3. Data preprocessing
# ---------------------------------------------------------------------------
print("Step 3: Preprocessing data...")

# Mean imputation for null values
for col in FEATURES:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"  Filled {col} nulls with mean = {mean_val:.2f}")

# Label encoding for the categorical target label (kept for reference /
# classification display; the regression itself predicts the continuous score)
le = LabelEncoder()
df["HDI_Category_Encoded"] = le.fit_transform(df["HDI_Category"])
print(f"  Label classes: {list(le.classes_)}")

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 4. Train/test split (75/25)
# ---------------------------------------------------------------------------
print("\nStep 4: Splitting data (75% train / 25% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"  Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 5. Train Linear Regression model
# ---------------------------------------------------------------------------
print("\nStep 5: Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("  Model trained.")
print(f"  Coefficients: {dict(zip(FEATURES, model.coef_))}")
print(f"  Intercept: {model.intercept_:.4f}")

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
print("\nStep 6: Evaluating model...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"  Train R^2 Score : {train_r2:.4f}")
print(f"  Test  R^2 Score : {test_r2:.4f}")
print(f"  Test  MAE       : {mae:.4f}")
print(f"  Test  RMSE      : {rmse:.4f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, edgecolor="k")
plt.plot([0, 1], [0, 1], "r--", label="Perfect Prediction")
plt.xlabel("Actual HDI Score")
plt.ylabel("Predicted HDI Score")
plt.title(f"Actual vs Predicted HDI (Test R2 = {test_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"))
plt.close()

metrics_summary = {
    "train_r2": round(train_r2, 4),
    "test_r2": round(test_r2, 4),
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
}
with open(os.path.join(BASE_DIR, "metrics.txt"), "w") as f:
    for k, v in metrics_summary.items():
        f.write(f"{k}: {v}\n")

# ---------------------------------------------------------------------------
# 7. Save model with pickle
# ---------------------------------------------------------------------------
print("\nStep 7: Saving model with pickle...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"  Model saved to {MODEL_PATH}")

print("\nPipeline complete.")

"""
generate_dataset.py
Generates a realistic synthetic Human Development Index (HDI) dataset
modeled on the official UNDP HDI methodology:

HDI = (Life Expectancy Index * Education Index * Income Index) ^ (1/3)

  Life Expectancy Index = (LE - 20) / (85 - 20)
  Education Index       = (MYS_index + EYS_index) / 2
      MYS_index = MYS / 15
      EYS_index = EYS / 18
  Income Index          = (log(GNI) - log(100)) / (log(75000) - log(100))

We build a list of ~160 country-like records with plausible ranges of
Life Expectancy, Mean Years of Schooling, Expected Years of Schooling and
GNI per Capita, compute the true HDI, then add a small amount of noise to
mimic real-world measurement/reporting variance.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

countries = [
    "Norway","Switzerland","Ireland","Germany","Hong Kong","Australia","Iceland",
    "Sweden","Singapore","Netherlands","Denmark","Finland","United Kingdom",
    "Belgium","New Zealand","Canada","United States","Japan","South Korea",
    "Israel","Luxembourg","France","Slovenia","Austria","Spain","Italy",
    "Czech Republic","Malta","Estonia","Greece","Cyprus","Poland","Lithuania",
    "Saudi Arabia","Bahrain","Portugal","Latvia","Chile","Hungary","Croatia",
    "United Arab Emirates","Slovakia","Qatar","Argentina","Montenegro","Romania",
    "Kuwait","Russia","Bulgaria","Uruguay","Bahamas","Kazakhstan","Belarus",
    "Turkey","Costa Rica","Mauritius","Serbia","Cuba","Panama","Georgia",
    "Malaysia","Barbados","Trinidad and Tobago","Mexico","Thailand","Brazil",
    "Colombia","China","Sri Lanka","Ecuador","Peru","Armenia","Ukraine",
    "Azerbaijan","Dominican Republic","Fiji","Jordan","Tunisia","Jamaica",
    "Algeria","Egypt","Indonesia","Vietnam","Philippines","Palestine","Bolivia",
    "South Africa","Iraq","Morocco","Guatemala","Nicaragua","Bangladesh",
    "India","Ghana","Kenya","Cambodia","Zambia","Myanmar","Nepal","Pakistan",
    "Angola","Congo","Tanzania","Nigeria","Rwanda","Cameroon","Uganda",
    "Senegal","Sudan","Haiti","Ethiopia","Malawi","Mozambique","Yemen",
    "Afghanistan","Niger","Chad","Mali","Burkina Faso","South Sudan",
    "Sierra Leone","Guinea","Burundi","Central African Republic","Somalia",
    "Liberia","Madagascar","DR Congo","Comoros","Gambia","Togo","Benin",
    "Zimbabwe","Lesotho","Eswatini","Botswana","Namibia","Gabon","Djibouti",
    "Mauritania","Papua New Guinea","Laos","Bhutan","Mongolia","Uzbekistan",
    "Kyrgyzstan","Tajikistan","Turkmenistan","Moldova","Albania","North Macedonia",
    "Bosnia and Herzegovina","El Salvador","Honduras","Paraguay","Venezuela",
    "Lebanon","Syria","Libya"
]

n = len(countries)

# Assign development tiers roughly matching real-world HDI rankings
# (0 = Very High, 1 = High, 2 = Medium, 3 = Low) so well-known countries
# land in a believable bracket instead of a purely random one.
very_high = set(countries[0:44])
high = set(countries[44:86])
medium = set(countries[86:120])
# everything else -> low

tier = []
for c in countries:
    if c in very_high:
        tier.append(0)
    elif c in high:
        tier.append(1)
    elif c in medium:
        tier.append(2)
    else:
        tier.append(3)
tier = np.array(tier)

life_expectancy = []
mean_schooling = []
expected_schooling = []
gni_per_capita = []

for t in tier:
    if t == 0:  # Very High
        life_expectancy.append(np.random.uniform(78, 85))
        mean_schooling.append(np.random.uniform(11.5, 14.5))
        expected_schooling.append(np.random.uniform(15, 18))
        gni_per_capita.append(np.random.uniform(30000, 90000))
    elif t == 1:  # High
        life_expectancy.append(np.random.uniform(72, 78))
        mean_schooling.append(np.random.uniform(8.5, 11.5))
        expected_schooling.append(np.random.uniform(12, 15))
        gni_per_capita.append(np.random.uniform(9000, 30000))
    elif t == 2:  # Medium
        life_expectancy.append(np.random.uniform(64, 72))
        mean_schooling.append(np.random.uniform(5, 8.5))
        expected_schooling.append(np.random.uniform(9, 12))
        gni_per_capita.append(np.random.uniform(2500, 9000))
    else:  # Low
        life_expectancy.append(np.random.uniform(50, 64))
        mean_schooling.append(np.random.uniform(1.5, 5))
        expected_schooling.append(np.random.uniform(5, 9))
        gni_per_capita.append(np.random.uniform(500, 2500))

life_expectancy = np.array(life_expectancy)
mean_schooling = np.array(mean_schooling)
expected_schooling = np.array(expected_schooling)
gni_per_capita = np.array(gni_per_capita)

# --- Official UNDP-style HDI formula ---
le_index = (life_expectancy - 20) / (85 - 20)
mys_index = np.clip(mean_schooling / 15, 0, 1)
eys_index = np.clip(expected_schooling / 18, 0, 1)
edu_index = (mys_index + eys_index) / 2
income_index = (np.log(gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100))
income_index = np.clip(income_index, 0, 1)

hdi = (le_index * edu_index * income_index) ** (1 / 3)

# small measurement noise
hdi = np.clip(hdi + np.random.normal(0, 0.01, size=n), 0, 1)

def classify(score):
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"

hdi_category = [classify(s) for s in hdi]

df = pd.DataFrame({
    "Country": countries,
    "Life_Expectancy": np.round(life_expectancy, 1),
    "Mean_Years_Schooling": np.round(mean_schooling, 2),
    "Expected_Years_Schooling": np.round(expected_schooling, 2),
    "GNI_Per_Capita": np.round(gni_per_capita, 0),
    "HDI_Score": np.round(hdi, 4),
    "HDI_Category": hdi_category,
})

# introduce a few missing values to make preprocessing meaningful
missing_idx = np.random.choice(df.index, size=6, replace=False)
for idx in missing_idx:
    col = np.random.choice(["Life_Expectancy", "Mean_Years_Schooling", "GNI_Per_Capita"])
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/hdi-predictor/data/hdi_dataset.csv", index=False)
print("Dataset generated:", df.shape)
print(df.head())

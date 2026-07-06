# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 7 : FEATURE IMPORTANCE
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# CREATE IMAGES FOLDER
# ==========================================================

os.makedirs("images", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
)

# Drop Customer ID
df.drop("Customer", axis=1, inplace=True)

# Date Features
df["Effective To Date"] = pd.to_datetime(
    df["Effective To Date"],
    format="%m/%d/%y"
)

df["Month"] = df["Effective To Date"].dt.month
df["Day"] = df["Effective To Date"].dt.day

df.drop("Effective To Date", axis=1, inplace=True)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Customer Lifetime Value", axis=1)
y = df["Customer Lifetime Value"]

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# ==========================================================
# PREPROCESSOR
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================================
# TRANSFORM DATA
# ==========================================================

X_processed = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_processed, y)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top20 = importance.head(20)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10,8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 20 Most Important Features")

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png",
    dpi=300
)

plt.show()

print("="*60)
print("Feature Importance Saved Successfully!")
print("="*60)

print(top20)
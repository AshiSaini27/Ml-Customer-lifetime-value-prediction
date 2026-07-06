# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# ADVANCED MODEL TRAINING
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# Load Dataset
# ==========================

print("="*70)
print("Loading Dataset...")
print("="*70)

df = pd.read_csv(
    "data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
)

print("\nDataset Loaded Successfully!")
print("Shape :", df.shape)

# ==========================
# Drop Unnecessary Column
# ==========================

df.drop("Customer", axis=1, inplace=True)

# ==========================
# Date Feature Engineering
# ==========================

df["Effective To Date"] = pd.to_datetime(
    df["Effective To Date"],
    format="%m/%d/%y"
)

df["Month"] = df["Effective To Date"].dt.month
df["Day"] = df["Effective To Date"].dt.day

df.drop("Effective To Date", axis=1, inplace=True)

print("\nDate Features Created Successfully!")

# ==========================
# Features and Target
# ==========================

X = df.drop("Customer Lifetime Value", axis=1)

y = df["Customer Lifetime Value"]

# ==========================
# Numerical & Categorical Columns
# ==========================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical Features")
print(numerical_features)

print("\nCategorical Features")
print(categorical_features)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

# Numerical Pipeline

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical Pipeline

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine Both Pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing Pipeline Created Successfully!")

# ==========================================================
# MODELS
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

print("\nModels Initialized Successfully!")


# ==========================================================
# TRAINING & MODEL COMPARISON
# ==========================================================

best_model = None
best_model_name = ""
best_r2 = float("-inf")

results = []

print("\n" + "=" * 70)
print("MODEL TRAINING STARTED")
print("=" * 70)

for model_name, model in models.items():

    # Create Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train Model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, mae, rmse, r2])

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² Score : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_model_name = model_name

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(f"Model : {best_model_name}")
print(f"R² Score : {best_r2:.4f}")


# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2 Score"
    ]
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully!")
print("Location : models/best_model.pkl")
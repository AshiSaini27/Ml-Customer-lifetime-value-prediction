# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 6 : HYPERPARAMETER TUNING
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

df = pd.read_csv(
    "data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
)

# Remove Customer ID
df.drop("Customer", axis=1, inplace=True)

# Date Feature Engineering
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
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# PREPROCESSING
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
# PIPELINE
# ==========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)

# ==========================================================
# PARAMETERS
# ==========================================================

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [10, 20, 30, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

# ==========================================================
# RANDOM SEARCH
# ==========================================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Optimized Random Forest...\n")

search.fit(X_train, y_train)

# ==========================================================
# RESULTS
# ==========================================================

best_model = search.best_estimator_

prediction = best_model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction) ** 0.5
r2 = r2_score(y_test, prediction)

print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(search.best_params_)

print("\nMAE :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R² Score :", round(r2, 4))

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(best_model, "models/tuned_model.pkl")

print("\nTuned model saved successfully!")
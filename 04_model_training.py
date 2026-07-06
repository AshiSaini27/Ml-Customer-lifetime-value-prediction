# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 4 : FEATURE ENGINEERING & MODEL TRAINING
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

# ==========================================================
# Data Preprocessing
# ==========================================================

# Remove Customer ID
df.drop("Customer", axis=1, inplace=True)

# Convert Date
df["Effective To Date"] = pd.to_datetime(
    df["Effective To Date"],
    format="%m/%d/%y"
)

df["Month"] = df["Effective To Date"].dt.month
df["Day"] = df["Effective To Date"].dt.day

df.drop("Effective To Date", axis=1, inplace=True)

# ==========================================================
# Label Encoding
# ==========================================================

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ==========================================================
# Features & Target
# ==========================================================

X = df.drop("Customer Lifetime Value", axis=1)
y = df["Customer Lifetime Value"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Models
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

best_model = None
best_score = -999

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** 0.5
    r2 = r2_score(y_test, prediction)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

print("\n" + "=" * 70)
print("Best Model Selected Successfully!")
print("=" * 70)

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully!")
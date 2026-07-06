# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 2 : DATA PREPROCESSING
# ==========================================================

import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

print("="*60)
print("Original Dataset Shape :", df.shape)
print("="*60)


# ==========================
# Drop Customer ID
# ==========================

df.drop("Customer", axis=1, inplace=True)

print("\nCustomer ID column removed.")


# ==========================
# Convert Date Column
# ==========================

df["Effective To Date"] = pd.to_datetime(df["Effective To Date"])

df["Month"] = df["Effective To Date"].dt.month
df["Day"] = df["Effective To Date"].dt.day

# Drop original date column
df.drop("Effective To Date", axis=1, inplace=True)

print("Date column converted into Month and Day.")


# ==========================
# Final Dataset Info
# ==========================

print("\nFinal Shape :", df.shape)

print("\nColumns\n")
print(df.columns)

print("\nData Types\n")
print(df.dtypes)

print("\nFirst Five Rows\n")
print(df.head())
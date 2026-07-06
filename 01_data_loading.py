# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 1 : DATA LOADING & EXPLORATION
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

print("=" * 60)
print("Customer Lifetime Value Prediction Project")
print("=" * 60)


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

print("\n✅ Dataset Loaded Successfully!\n")


# ==========================
# Basic Exploration
# ==========================

print("First 5 Rows")
print(df.head())

print("\n" + "="*60)

print("Dataset Shape")
print(df.shape)

print("\n" + "="*60)

print("Column Names")
print(df.columns)

print("\n" + "="*60)

print("Dataset Info")
print(df.info())

print("\n" + "="*60)

print("Missing Values")
print(df.isnull().sum())

print("\n" + "="*60)

print("Duplicate Rows :", df.duplicated().sum())

print("\n" + "="*60)

print("Statistical Summary")
print(df.describe(include="all"))
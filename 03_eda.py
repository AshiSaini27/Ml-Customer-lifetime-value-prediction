# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION
# STEP 3 : EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

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

print("=" * 60)
print("Generating EDA Plots...")
print("=" * 60)

# ==========================================================
# Distribution of Target Variable
# ==========================================================

plt.figure(figsize=(8,5))
sns.histplot(df["Customer Lifetime Value"], bins=40, kde=True)
plt.title("Customer Lifetime Value Distribution")
plt.tight_layout()
plt.savefig("images/clv_distribution.png")
plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include=["number"]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()

# ==========================================================
# Income Distribution
# ==========================================================

plt.figure(figsize=(8,5))
sns.histplot(df["Income"], bins=40)
plt.title("Income Distribution")
plt.tight_layout()
plt.savefig("images/income_distribution.png")
plt.show()

# ==========================================================
# Vehicle Class
# ==========================================================

plt.figure(figsize=(8,5))
sns.countplot(x=df["Vehicle Class"])
plt.xticks(rotation=20)
plt.title("Vehicle Class")
plt.tight_layout()
plt.savefig("images/vehicle_class.png")
plt.show()

# ==========================================================
# Coverage
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(x=df["Coverage"])
plt.title("Coverage")
plt.tight_layout()
plt.savefig("images/coverage.png")
plt.show()

print("\n✅ All EDA graphs saved inside images folder.")
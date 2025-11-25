import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------
# 1. Load the COMPAS Two-Year Dataset
# -------------------------------------
df = pd.read_csv("compas-scores-two-years.csv")

# -------------------------------------
# 2. Clean & Prepare Data
# -------------------------------------
# Normalize race column
df["race"] = df["race"].str.lower().str.strip()

# Ensure decile_score exists and is numeric
df["decile_score"] = pd.to_numeric(df["decile_score"], errors="coerce")

# Drop rows missing race or decile score
df = df.dropna(subset=["race", "decile_score"])

# -------------------------------------
# 3. Define High-Risk (Standard COMPAS Method)
# -------------------------------------
df["high_risk"] = df["decile_score"] >= 5

# -------------------------------------
# 4. Compute Selection Rate by Race
# -------------------------------------
selection_rate = df.groupby("race")["high_risk"].mean()

print("\nSelection Rate by Race:")
print(selection_rate)

# -------------------------------------
# 5. Plot Selection Rate
# -------------------------------------
plt.figure(figsize=(10, 6))
selection_rate.plot(kind="bar")

plt.title("Selection Rate (High Risk Classification) by Race")
plt.xlabel("Race")
plt.ylabel("Selection Rate")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

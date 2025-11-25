import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataframe here (adjust the file path as needed)
df = pd.read_csv("fairface_results.csv")

results = df[df["predicted_race"].notnull() & (df["predicted_race"] != "error")]

race_groups = results["race"].unique()

metrics = []

for race in race_groups:
    subset = results[results["race"] == race]
    accuracy = (subset["race"] == subset["predicted_race"]).mean()
    selection_rate = len(subset) / len(results)

    metrics.append({
        "race": race,
        "accuracy": accuracy,
        "selection_rate": selection_rate
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df)
plt.figure(figsize=(10,5))
sns.barplot(x="race", y="accuracy", data=metrics_df)
plt.title("Accuracy by Race (Bias Indicator)")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x="race", y="selection_rate", data=metrics_df)
plt.title("Selection Rate by Race")
plt.xticks(rotation=45)
plt.show()

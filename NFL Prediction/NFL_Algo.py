import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# ===== CONFIG =====
DATA_DIR = r"C:\Users\casey\OneDrive\Desktop\Personal\Coding\NFL Prediction\data"
TRAIN_YEARS = [2022, 2023, 2024]
TARGET_YEAR = 2025
FEATURE_COLS = ["Vegas Rank", "QB Rank", "Coach Rank", "OL Rank", "D Rank", "SOS Rank"]
COL_TEAM, COL_FINAL = "Team", "Final Rank"
# ==================

# Load data
dfs = {y: pd.read_csv(os.path.join(DATA_DIR, f"{y}.csv")) for y in TRAIN_YEARS + [TARGET_YEAR]}
train_df = pd.concat([dfs[y].assign(Year=y) for y in TRAIN_YEARS], ignore_index=True)

X = train_df[FEATURE_COLS]
y = train_df[COL_FINAL]

# Fit
model = LinearRegression().fit(X, y)
print("\n=== Regression Coefficients ===")
for feat, wt in zip(FEATURE_COLS, model.coef_):
    print(f"{feat:>12s}: {wt:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# Predict 2025
df25 = dfs[TARGET_YEAR].copy()
df25["Score_reg"] = model.predict(df25[FEATURE_COLS])
df25["PredictedRank_reg"] = df25["Score_reg"].rank(method="min", ascending=True).astype(int)

# Save
out = df25[[COL_TEAM, "Score_reg", "PredictedRank_reg"]].sort_values("PredictedRank_reg")
out_path = os.path.join(DATA_DIR, f"rankings_{TARGET_YEAR}_regression.csv")
out.to_csv(out_path, index=False)
print(f"\n[OK] Saved: {out_path}")


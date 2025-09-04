import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ===== CONFIG =====
DATA_DIR = r"C:\Users\casey\OneDrive\Desktop\Personal\Coding\NFL Prediction\data"
TRAIN_YEARS = [2022, 2023, 2024]
TARGET_YEAR = 2025
FEATURE_COLS = ["Vegas Rank", "QB Rank", "Coach Rank", "OL Rank", "D Rank", "SOS Rank"]
COL_TEAM, COL_FINAL = "Team", "Final Rank"
ITERATIONS = 50000  # speed/quality tradeoff
RANDOM_SEED = 42
# ==================

np.random.seed(RANDOM_SEED)

# Load data
dfs = {y: pd.read_csv(os.path.join(DATA_DIR, f"{y}.csv")) for y in TRAIN_YEARS + [TARGET_YEAR]}
train_df = pd.concat([dfs[y].assign(Year=y) for y in TRAIN_YEARS], ignore_index=True)

def evaluate(weights: np.ndarray):
    scores = (train_df[FEATURE_COLS].values * weights).sum(axis=1)
    # convert to rank order (lower score => better final rank prediction)
    pred_rank = scores.argsort().argsort() + 1
    rho, _ = spearmanr(pred_rank, train_df[COL_FINAL].values)
    mae = np.mean(np.abs(pred_rank - train_df[COL_FINAL].values))
    return rho, mae

best = None
for _ in range(ITERATIONS):
    w = np.random.rand(len(FEATURE_COLS))  # random positive weights
    rho, mae = evaluate(w)
    if not best or (rho > best["rho"]) or (np.isclose(rho, best["rho"]) and mae < best["mae"]):
        best = {"w": w, "rho": rho, "mae": mae}

print("\n=== Simple Model Weights (random search) ===")
for feat, wt in zip(FEATURE_COLS, best["w"]):
    print(f"{feat:>12s}: {wt:.3f}")
print(f"Spearman: {best['rho']:.3f} | MAE: {best['mae']:.2f}")

# Predict 2025
df25 = dfs[TARGET_YEAR].copy()
df25["Score_simple"] = (df25[FEATURE_COLS].values * best["w"]).sum(axis=1)
df25["PredictedRank_simple"] = df25["Score_simple"].rank(method="min", ascending=True).astype(int)

# Save
out = df25[[COL_TEAM, "Score_simple", "PredictedRank_simple"]].sort_values("PredictedRank_simple")
out_path = os.path.join(DATA_DIR, f"rankings_{TARGET_YEAR}_simple.csv")
out.to_csv(out_path, index=False)
print(f"[OK] Saved: {out_path}")

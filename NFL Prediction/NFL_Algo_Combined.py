import os
import pandas as pd
import numpy as np

# ===== CONFIG =====
DATA_DIR = r"C:\Users\casey\OneDrive\Desktop\Personal\Coding\NFL Prediction\data"
YEAR = 2025
FILE_REG = os.path.join(DATA_DIR, f"rankings_{YEAR}_regression.csv")  # from NFL_Algo_regression.py
FILE_SIM = os.path.join(DATA_DIR, f"rankings_{YEAR}_simple.csv")      # from NFL_Algo_simple.py

COL_TEAM = "Team"
COL_SCORE_REG = "Score_reg"
COL_RANK_REG  = "PredictedRank_reg"
COL_SCORE_SIM = "Score_simple"
COL_RANK_SIM  = "PredictedRank_simple"

# If True, z-normalize each model's scores before averaging (recommended)
NORMALIZE_SCORES = True
# ==================

def zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s != 0 else x * 0.0

def main():
    reg = pd.read_csv(FILE_REG)  # expects Team, Score_reg, PredictedRank_reg
    sim = pd.read_csv(FILE_SIM)  # expects Team, Score_simple, PredictedRank_simple

    # Merge on team
    df = pd.merge(reg, sim, on=COL_TEAM, how="inner")

    # --- Average SCORE ensemble ---
    if NORMALIZE_SCORES:
        # normalize each score column to mean=0, std=1 before averaging
        df["Score_reg_norm"] = zscore(df[COL_SCORE_REG].values)
        df["Score_simple_norm"] = zscore(df[COL_SCORE_SIM].values)
        df["AvgScore"] = (df["Score_reg_norm"] + df["Score_simple_norm"]) / 2.0
    else:
        # raw average (only safe if both scores live on similar scales)
        df["AvgScore"] = (df[COL_SCORE_REG] + df[COL_SCORE_SIM]) / 2.0

    # lower AvgScore => better (consistent with earlier scripts)
    df["EnsembleRank_from_AvgScore"] = df["AvgScore"].rank(method="min", ascending=True).astype(int)

    # --- Average RANK ensemble ---
    df["AvgRank"] = (df[COL_RANK_REG] + df[COL_RANK_SIM]) / 2.0
    # We need to convert averaged ranks back to a final order
    df["EnsembleRank_from_AvgRank"] = df["AvgRank"].rank(method="min", ascending=True).astype(int)

    # Arrange nice output
    out_cols = [
        COL_TEAM,
        COL_SCORE_REG, COL_RANK_REG,
        COL_SCORE_SIM, COL_RANK_SIM,
        "AvgScore", "EnsembleRank_from_AvgScore",
        "AvgRank",  "EnsembleRank_from_AvgRank"
    ]
    if NORMALIZE_SCORES:
        out_cols[2:2] = ["Score_reg_norm"]   # insert after Score_reg
        out_cols[4:4] = ["Score_simple_norm"]  # insert after Score_simple

    out = df[out_cols].sort_values("EnsembleRank_from_AvgScore").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, f"rankings_{YEAR}_ensemble2.csv")
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved ensemble to: {out_path}")
    print("\n=== Top by AvgScore ensemble ===")
    print(out[[COL_TEAM, "AvgScore", "EnsembleRank_from_AvgScore"]].head(10).to_string(index=False))

    print("\n=== Top by AvgRank ensemble ===")
    print(out[[COL_TEAM, "AvgRank", "EnsembleRank_from_AvgRank"]].sort_values("EnsembleRank_from_AvgRank").head(10).to_string(index=False))

if __name__ == "__main__":
    main()

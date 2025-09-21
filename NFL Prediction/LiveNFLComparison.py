# LiveNFLComparison.py
import os, re, requests, pandas as pd, numpy as np

# ===== CONFIG =====
DATA_DIR = r"C:\Users\casey\OneDrive\Desktop\Personal\Coding\NFL Prediction\data"
YEAR = 2025
ENSEMBLE_FILE = os.path.join(DATA_DIR, f"rankings_{YEAR}_ensemble2.csv")
OUT_FILE      = os.path.join(DATA_DIR, f"LiveNFLComparison{YEAR}.csv")

ESPN_PRIMARY  = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings?region=us"
ESPN_FALLBACK = "https://cdn.espn.com/core/nfl/standings?xhr=1&group=league"

EXPECTED_RANK_COLS = ["EnsembleRank_from_AvgScore"]
# ==================

def normalize_cols(df):
    def clean(s):
        s = str(s).replace("\u00A0"," ")
        s = re.sub(r"\s+"," ", s)
        return s.strip()
    df.columns = [clean(c) for c in df.columns]
    return df

def clean_team_name(name: str) -> str:
    name = re.sub(r"[\*\u2020\u2713#\+]+", "", str(name)).strip()
    fixes = {
        "Washington Football Team": "Washington Commanders",
        "Washington Redskins": "Washington Commanders",
        "LA Chargers": "Los Angeles Chargers",
        "LA Rams": "Los Angeles Rams",
        "Oakland Raiders": "Las Vegas Raiders",
        "St. Louis Rams": "Los Angeles Rams",
        "San Diego Chargers": "Los Angeles Chargers",
    }
    return fixes.get(name, name)

def _flatten(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten(v)

def _entries_to_team_list(entries):
    teams, seen = [], set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        t = e.get("team", {})
        name = (
            (isinstance(t, dict) and (t.get("displayName") or t.get("name") or t.get("location") or t.get("shortDisplayName")))
            or e.get("displayName") or e.get("name")
        )
        if name:
            name = clean_team_name(str(name))
            if name and name not in seen:
                seen.add(name)
                teams.append(name)
    return teams

def fetch_espn_league_order() -> pd.DataFrame:
    headers = {"User-Agent":"Mozilla/5.0"}
    last_err = None
    for url in (ESPN_PRIMARY, ESPN_FALLBACK):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            last_err = e
            continue

        candidates = []
        for node in _flatten(data):
            if isinstance(node, dict):
                entries = node.get("entries") or (node.get("standings", {}).get("entries") if isinstance(node.get("standings"), dict) else None)
                if isinstance(entries, list) and len(entries) >= 32:
                    teams = _entries_to_team_list(entries)
                    if len(teams) == 32:
                        candidates.append(teams)
            if isinstance(node, list) and len(node) >= 32:
                teams = _entries_to_team_list(node)
                if len(teams) == 32:
                    candidates.append(teams)

        if candidates:
            teams = candidates[0]  # ESPN order top->bottom (1 = best)
            df = pd.DataFrame({"Team": teams})
            df["Live Rank"] = np.arange(1, 33)
            return df

    raise RuntimeError(f"Could not locate a 32-team league array in ESPN JSON. Last error: {last_err}")

def main():
    print(f"[INFO] Reading ensemble:\n  {ENSEMBLE_FILE}")
    ens = pd.read_csv(ENSEMBLE_FILE, encoding="utf-8")
    ens = normalize_cols(ens)
    ens["Team"] = ens["Team"].apply(clean_team_name)
    print("[INFO] Ensemble columns:", list(ens.columns))

    # Expect just AvgScore ensemble rank in your file
    if "EnsembleRank_from_AvgScore" not in ens.columns:
        raise ValueError("Expected 'EnsembleRank_from_AvgScore' in your CSV.")

    # Flip your ensemble rank IN-PLACE so 1 = best (was 32 = best)
    ens_small = ens[["Team", "EnsembleRank_from_AvgScore"]].copy()
    ens_small["EnsembleRank_from_AvgScore"] = 33 - ens_small["EnsembleRank_from_AvgScore"]

    # Live (ESPN league-wide order; already 1 = best)
    live = fetch_espn_league_order()
    print("[INFO] Pulled ESPN league order successfully.")

    out = (live.merge(ens_small, on="Team", how="inner")
               .sort_values("Live Rank")
               .reset_index(drop=True))

    # Print positions-only view (both are 1 = best)
    print("\n=== Live (ESPN order) vs Ensemble (positions only; 1 = best) ===")
    print(out[["Team", "Live Rank", "EnsembleRank_from_AvgScore"]].to_string(index=False))

    # Save ONLY these three columns (no temporary 'best' column)
    out[["Team", "Live Rank", "EnsembleRank_from_AvgScore"]].to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()








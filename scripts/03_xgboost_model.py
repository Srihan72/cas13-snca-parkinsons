"""
03_xgboost_model.py
-------------------
Train an XGBoost regression model on the Wessels et al. 2020 HEK293T guide
efficacy data and apply it to all 2500 SNCA gRNA candidates.

Training data:  data/41587_2020_456_MOESM3_ESM.xlsx  (HEK293T_screen sheet)
  - GuideSeq                  : 27-nt spacer sequence
  - standardizedGuideScores   : normalized knockdown efficiency (0–1),
                                 derived from the GFP tiling screen model
                                 and validated on HEK293T essential-gene data

Features engineered per spacer:
  gc          – GC fraction
  A/U/G/C     – mono-nucleotide frequencies (DNA T treated as RNA U)
  AA…UU       – 16 di-nucleotide frequencies
  norm_pos    – start position in transcript, normalized to [0, 1]
  mfe         – RNAfold minimum free energy of the 50-nt target window
  accessibility – fraction of unpaired bases in the 23-nt target region

Outputs:
  models/xgboost_efficiency.json
  output/grna_candidates_predicted.csv
"""

import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
WESSELS      = BASE / "data"   / "41587_2020_456_MOESM3_ESM.xlsx"
SCORED_CSV   = BASE / "output" / "grna_candidates_scored.csv"
OUT_CSV      = BASE / "output" / "grna_candidates_predicted.csv"
MODEL_PATH   = BASE / "models" / "xgboost_efficiency.json"
MODEL_PATH.parent.mkdir(exist_ok=True)

SNCA_LEN = 3177   # NM_000345.4

# ── Feature vocabulary ────────────────────────────────────────────────────────
_NTS  = list("ACGU")
_DINS = ["".join(p) for p in product(_NTS, repeat=2)]

BASE_FEAT_COLS = ["gc"] + _NTS + _DINS + ["norm_pos"]
ALL_FEAT_COLS  = BASE_FEAT_COLS + ["mfe", "accessibility"]


# ── Feature helpers ───────────────────────────────────────────────────────────
def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def engineer_features(seq: str, norm_pos: float) -> dict:
    s = to_rna(seq)
    n = len(s)

    feats: dict = {}
    feats["gc"] = (s.count("G") + s.count("C")) / n

    for nt in _NTS:
        feats[nt] = s.count(nt) / n

    n_di = n - 1
    di_counts = {d: 0 for d in _DINS}
    for i in range(n_di):
        di = s[i : i + 2]
        if di in di_counts:
            di_counts[di] += 1
    for d, c in di_counts.items():
        feats[d] = c / n_di

    feats["norm_pos"] = norm_pos
    return feats


def extract_pos_from_guide_name(name: str) -> float:
    """Parse start position from 'GENE_Bk_crRNAnnnn:start-end' format."""
    m = re.search(r":(\d+)-\d+", name)
    return float(m.group(1)) if m else np.nan


# ── Main ─────────────────────────────────────────────────────────────────────
def main():

    # ── 1. Load & explore Wessels ─────────────────────────────────────────────
    print("=" * 65)
    print("WESSELS ET AL. 2020  –  HEK293T_screen")
    print("=" * 65)

    raw = pd.read_excel(WESSELS, sheet_name="HEK293T_screen", skiprows=14)

    print(f"\nColumns ({len(raw.columns)}):\n  {raw.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(raw.head(3).to_string())
    print(f"\nShape           : {raw.shape}")
    print(f"Class dist      : {raw['Class'].value_counts().to_dict()}")
    print(f"GuideSeq lengths: {raw['GuideSeq'].dropna().str.len().unique().tolist()} nt")
    print(f"Score range     : {raw['standardizedGuideScores'].min():.4f} – "
          f"{raw['standardizedGuideScores'].max():.4f}")

    df = raw.dropna(subset=["GuideSeq", "standardizedGuideScores"]).copy()
    print(f"\nRows with sequence + score: {len(df)}")

    # Position: extract from GuideName, normalise per gene by max observed pos
    df["_pos"] = df["GuideName"].apply(extract_pos_from_guide_name)
    gene_max   = df.groupby("Gene")["_pos"].transform("max").replace(0, np.nan)
    df["norm_pos"] = (df["_pos"] / gene_max).fillna(0.5)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\nEngineering sequence features…")
    feat_df = pd.DataFrame(
        [engineer_features(seq, np) for seq, np in zip(df["GuideSeq"], df["norm_pos"])],
        index=df.index,
        columns=BASE_FEAT_COLS,
    )

    # ── 3. Join MFE & accessibility where SNCA sequences match ───────────────
    scored       = pd.read_csv(SCORED_CSV)
    struct_lookup = (
        scored.set_index("spacer")[["mfe", "accessibility"]]
        .to_dict("index")
    )

    def _lookup(seq, field):
        key = seq.upper().replace("U", "T")
        return struct_lookup.get(key, {}).get(field, np.nan)

    feat_df["mfe"]           = [_lookup(s, "mfe")           for s in df["GuideSeq"]]
    feat_df["accessibility"] = [_lookup(s, "accessibility") for s in df["GuideSeq"]]

    n_matched = feat_df["mfe"].notna().sum()
    print(f"Wessels→SNCA sequence matches: {n_matched}/{len(df)}  "
          f"(expected ~0; MFE/accessibility will be NaN for training rows)")

    X      = feat_df[ALL_FEAT_COLS].values.astype(float)
    y      = df["standardizedGuideScores"].values.astype(float)

    print(f"\nFeature matrix : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target range   : {y.min():.4f} – {y.max():.4f}  "
          f"(mean {y.mean():.4f})")

    # ── 4. 5-fold cross-validation ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 65)

    xgb_params = dict(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        tree_method      = "hist",
        verbosity        = 0,
    )

    kf        = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list, mae_list = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[val_idx])
        r2  = r2_score(y[val_idx], preds)
        mae = mean_absolute_error(y[val_idx], preds)
        r2_list.append(r2)
        mae_list.append(mae)
        print(f"  Fold {fold}:  R² = {r2:+.4f}   MAE = {mae:.4f}")

    print(f"\n  Mean :  R² = {np.mean(r2_list):+.4f} ± {np.std(r2_list):.4f}"
          f"   MAE = {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")

    # ── 5. Train on full data & save ─────────────────────────────────────────
    print("\nRetraining on full dataset…")
    final = xgb.XGBRegressor(**xgb_params)
    final.fit(X, y)
    final.save_model(str(MODEL_PATH))
    print(f"Model saved  → {MODEL_PATH}")

    # Feature importance (top 10)
    importances = pd.Series(
        final.feature_importances_, index=ALL_FEAT_COLS
    ).sort_values(ascending=False)
    print(f"\nTop 10 feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<15} {imp:.4f}")

    # ── 6. Apply model to all 2500 SNCA candidates ────────────────────────────
    print("\n" + "=" * 65)
    print("APPLYING MODEL TO 2500 SNCA CANDIDATES")
    print("=" * 65)

    scored["norm_pos"] = scored["start"] / SNCA_LEN

    snca_base = pd.DataFrame(
        [engineer_features(seq, np_)
         for seq, np_ in zip(scored["spacer"], scored["norm_pos"])],
        columns=BASE_FEAT_COLS,
    )
    snca_base["mfe"]           = scored["mfe"].values
    snca_base["accessibility"] = scored["accessibility"].values

    snca_X = snca_base[ALL_FEAT_COLS].values.astype(float)

    scored["predicted_efficiency"] = final.predict(snca_X).clip(0, 1)
    scored.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(scored)} predictions → {OUT_CSV}")

    # ── 7. Top 10 candidates ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TOP 10 SNCA gRNA CANDIDATES BY PREDICTED KNOCKDOWN EFFICIENCY")
    print("=" * 65)

    top10 = (
        scored
        .nlargest(10, "predicted_efficiency")
        [["spacer", "strand", "start", "gc", "mfe", "accessibility",
          "predicted_efficiency"]]
        .copy()
        .reset_index(drop=True)
    )
    top10.index += 1
    top10["gc"]                   = (top10["gc"] * 100).round(1).astype(str) + "%"
    top10["mfe"]                  = top10["mfe"].map("{:.2f}".format)
    top10["accessibility"]        = top10["accessibility"].map("{:.3f}".format)
    top10["predicted_efficiency"] = top10["predicted_efficiency"].map("{:.4f}".format)

    print(top10.to_string())


if __name__ == "__main__":
    main()

"""
train.py — Clasificador de dirección USDT/VES (LightGBM + Walk-Forward)
========================================================================
Corre semanalmente en GitHub Actions (train.yml) o manual con
"workflow_dispatch". También funciona local si clonas el repo.

Uso:
    python train.py                     # dirección a T+15 min
    python train.py --horizon-min 30
"""

import argparse

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

from features import FEATURE_COLS, build_dataset

MODEL_PATH = "model.pkl"
MIN_SAMPLES = 500

TO_LGB = {-1: 0, 0: 1, 1: 2}
FROM_LGB = {v: k for k, v in TO_LGB.items()}

PARAMS = dict(
    objective="multiclass", num_class=3,
    n_estimators=400, learning_rate=0.05,
    max_depth=5, num_leaves=31, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    verbosity=-1,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    X, y, _ = build_dataset(horizon_min=args.horizon_min,
                            threshold_pct=args.threshold)
    n = len(X)
    print(f"Dataset: {n} muestras, {len(FEATURE_COLS)} features")
    print(y.value_counts(normalize=True).sort_index().round(3).to_string())

    if n < 150:
        print(f"\n[ABORT] Solo {n} muestras: aún no hay datos suficientes.")
        return
    if n < MIN_SAMPLES:
        print(f"\n[AVISO] {n} < {MIN_SAMPLES} muestras: modelo preliminar.")

    y_lgb = y.map(TO_LGB)
    tscv = TimeSeriesSplit(n_splits=args.folds)
    accs, bases, all_true, all_pred = [], [], [], []

    for i, (tr, te) in enumerate(tscv.split(X), 1):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X.iloc[tr], y_lgb.iloc[tr])
        pred = m.predict(X.iloc[te])
        acc = accuracy_score(y_lgb.iloc[te], pred)
        base = (y_lgb.iloc[te] == y_lgb.iloc[tr].mode()[0]).mean()
        accs.append(acc); bases.append(base)
        all_true.extend(y_lgb.iloc[te]); all_pred.extend(pred)
        print(f"Fold {i}: accuracy={acc:.3f}  baseline={base:.3f}")

    print(f"\nWalk-forward: {np.mean(accs):.3f} vs baseline {np.mean(bases):.3f}")
    print("Si no supera al baseline, el modelo aún no aprende nada útil.\n")

    names = ["Baja (-1)", "Lateral (0)", "Sube (+1)"]
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
    print(" " * 14 + "".join(f"{t:>14}" for t in names))
    for t, r in zip(names, cm):
        print(f"{t:>14}" + "".join(f"{v:>14}" for v in r))
    print()
    print(classification_report(all_true, all_pred, labels=[0, 1, 2],
                                target_names=names, zero_division=0))

    final = lgb.LGBMClassifier(**PARAMS)
    final.fit(X, y_lgb)
    for name, v in sorted(zip(FEATURE_COLS, final.feature_importances_),
                          key=lambda t: -t[1])[:10]:
        print(f"  {name:<16} {v}")

    joblib.dump({
        "model": final, "feature_cols": FEATURE_COLS,
        "horizon_min": args.horizon_min, "threshold_pct": args.threshold,
        "from_lgb": FROM_LGB,
        "wf_accuracy": float(np.mean(accs)),
        "baseline_accuracy": float(np.mean(bases)),
    }, MODEL_PATH)
    print(f"\n[OK] Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()

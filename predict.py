"""
predict.py — Inferencia en vivo → prediction.json
==================================================
Corre en el mismo workflow que el collector (si existe model.pkl).
El JSON queda commiteado en el repo y tu dashboard lo lee con:

  fetch('https://raw.githubusercontent.com/USUARIO/REPO/main/prediction.json')
"""

import datetime as dt
import json

import joblib

from features import latest_feature_row

MODEL_PATH = "model.pkl"
OUT_PATH = "prediction.json"

SIGNAL_NAMES = {-1: "BAJA", 0: "LATERAL", 1: "SUBE"}
PROB_KEYS = {0: "baja", 1: "lateral", 2: "sube"}


def main():
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["feature_cols"]

    row = latest_feature_row()
    probs = model.predict_proba(row[cols])[0]
    best = int(probs.argmax())
    signal = bundle["from_lgb"][best]

    result = {
        "timestamp": dt.datetime.now(dt.timezone.utc)
                       .replace(microsecond=0).isoformat(),
        "snapshot_ts": row.index[0].isoformat(),
        "mid": round(float(row["mid"].iloc[0]), 4),
        "brecha_pct": round(float(row["brecha"].iloc[0]), 2),
        "imbalance": round(float(row["imbalance"].iloc[0]), 3),
        "rsi_14": round(float(row["rsi_14"].iloc[0]), 1),
        "horizon_min": bundle["horizon_min"],
        "threshold_pct": bundle["threshold_pct"],
        "probs": {PROB_KEYS[i]: round(float(p), 3)
                  for i, p in enumerate(probs)},
        "signal": SIGNAL_NAMES[signal],
        "confidence": round(float(probs[best]), 3),
        "model_wf_accuracy": round(bundle.get("wf_accuracy", 0), 3),
        "baseline_accuracy": round(bundle.get("baseline_accuracy", 0), 3),
        "disclaimer": "Estimación estadística, no asesoría financiera.",
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] {result['signal']} ({result['confidence']:.0%}) "
          f"a {result['horizon_min']} min")


if __name__ == "__main__":
    main()

"""Generate the bundled synthetic monitoring sample in e23mon/sample/.

A synthetic credit-card portfolio is split into a training set, a reference
set and six monthly slices. The last three slices carry injected covariate,
label and calibration drift; the July slice also contains an education code
never seen in the reference.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e23mon.data import DRIFTS, TARGET, score, synthetic_credit, train_models  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "e23mon" / "sample"
PLAN = [
    ("2026-04", None, 0.0),
    ("2026-05", None, 0.0),
    ("2026-06", None, 0.0),
    ("2026-07", "covariate", 0.5),
    ("2026-08", "label", 0.4),
    ("2026-09", "calibration", 0.08),
]


def main():
    rng = np.random.default_rng(2027)
    pop = synthetic_credit(40_000, seed=23)
    features = [c for c in pop.columns if c != TARGET]
    models = train_models(pop.iloc[:8000], features, seed=0)
    scored = score(models, pop.iloc[8000:]).reset_index(drop=True)
    scored[["pd_logit", "pd_gbm"]] = scored[["pd_logit", "pd_gbm"]].round(6)
    OUT.mkdir(parents=True, exist_ok=True)
    scored.iloc[:3000].to_csv(OUT / "reference.csv", index=False)
    pool = scored.iloc[3000:].reset_index(drop=True)
    slices = []
    for period, kind, delta in PLAN:
        if kind is None:
            s = pool.iloc[rng.integers(0, len(pool), 1200)].reset_index(drop=True)
        else:
            s = DRIFTS[kind](pool, 1200, delta, rng)
        if kind == "covariate":
            s.loc[s.index[:15], "EDUCATION"] = 7
        fname = f"slice_{period}.csv"
        s.to_csv(OUT / fname, index=False)
        slices.append({"period": period, "file": fname})
    cfg = {
        "reference": "reference.csv",
        "target_col": TARGET,
        "features": features,
        "models": [
            {"model_id": "CR-PD-001", "name": "Credit card PD scorecard (logistic)", "score_col": "pd_logit"},
            {"model_id": "CR-PD-002", "name": "Credit card PD challenger (gradient boosting)", "score_col": "pd_gbm"},
        ],
        "slices": slices,
    }
    (OUT / "monitoring.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote sample to {OUT}")


if __name__ == "__main__":
    main()

"""Model inventory: loading, rating rules and the validator.

The rules encode an E-23 style risk-based approach: each model gets a risk
rating from its tiering criteria, and the rating sets the maximum time between
independent reviews and between monitoring runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

RATINGS = ("High", "Medium", "Low")

# Rating -> maximum months between reviews and between monitoring runs.
RATING_RULES = {
    "High": {"review_months": 12, "monitoring_months": 1},
    "Medium": {"review_months": 24, "monitoring_months": 3},
    "Low": {"review_months": 36, "monitoring_months": 12},
}

# Tiering criteria. Score = 2 * materiality + complexity + AI/ML + regulatory use.
LEVEL_POINTS = {"High": 3, "Medium": 2, "Low": 1}
HIGH_CUTOFF = 8
MEDIUM_CUTOFF = 5

COLUMNS = [
    "model_id", "name", "owner", "business_use", "category", "materiality", "complexity",
    "uses_ai_ml", "regulatory_use", "risk_rating", "status", "last_review", "next_review",
    "monitoring_interval_months", "last_monitoring", "decommissioned_date", "last_scored_date",
]
DATE_COLUMNS = ["last_review", "next_review", "last_monitoring", "decommissioned_date", "last_scored_date", "as_of"]
ACTIVE = "Active"
DECOMMISSIONED = "Decommissioned"


@dataclass
class Violation:
    model_id: str
    rule: str
    detail: str


def load_inventory(path: str | Path) -> pd.DataFrame:
    """Read an inventory from CSV or YAML (a list under ``models:``)."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path) as f:
            doc = yaml.safe_load(f)
        df = pd.DataFrame(doc["models"] if isinstance(doc, dict) else doc)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].replace("", None), errors="coerce").dt.date
    for c in df.columns:
        if c not in DATE_COLUMNS:
            df[c] = df[c].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip())
    return df


def _flag(v) -> bool:
    return str(v).strip().lower() in ("yes", "y", "true", "1")


def criteria_score(materiality: str, complexity: str, uses_ai_ml, regulatory_use) -> int | None:
    m = LEVEL_POINTS.get(str(materiality).strip().title())
    c = LEVEL_POINTS.get(str(complexity).strip().title())
    if m is None or c is None:
        return None
    return 2 * m + c + int(_flag(uses_ai_ml)) + int(_flag(regulatory_use))


def expected_rating(materiality: str, complexity: str, uses_ai_ml=False, regulatory_use=False) -> str | None:
    s = criteria_score(materiality, complexity, uses_ai_ml, regulatory_use)
    if s is None:
        return None
    if s >= HIGH_CUTOFF:
        return "High"
    if s >= MEDIUM_CUTOFF:
        return "Medium"
    return "Low"


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(y, m, day)
        except ValueError:
            continue
    raise ValueError(d)


def _missing(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT


def resolve_as_of(df: pd.DataFrame, as_of: date | None) -> date:
    """Explicit date, else the inventory snapshot's ``as_of`` column, else today."""
    if as_of is not None:
        return as_of
    if "as_of" in df.columns:
        vals = [v for v in df["as_of"] if not _missing(v)]
        if vals:
            return max(vals)
    return date.today()


def validate(df: pd.DataFrame, as_of: date | None = None) -> list[Violation]:
    as_of = resolve_as_of(df, as_of)
    out: list[Violation] = []
    dup = df["model_id"][df["model_id"].duplicated(keep=False) & (df["model_id"] != "")]
    for mid in sorted(set(dup)):
        out.append(Violation(mid, "DUPLICATE_ID", "model_id appears more than once"))
    for _, r in df.iterrows():
        mid = r["model_id"] or "(no id)"
        status = r["status"].title()
        active = status == ACTIVE
        if not r["model_id"]:
            out.append(Violation(mid, "MISSING_ID", "model has no identifier"))
        if not r["owner"]:
            out.append(Violation(mid, "MISSING_OWNER", "no accountable model owner recorded"))
        rating = r["risk_rating"].title()
        if rating not in RATINGS:
            out.append(Violation(mid, "INVALID_RATING", f"risk rating '{r['risk_rating']}' is not one of {', '.join(RATINGS)}"))
            rule = None
        else:
            rule = RATING_RULES[rating]
        exp = expected_rating(r["materiality"], r["complexity"], r["uses_ai_ml"], r["regulatory_use"])
        if exp is None:
            out.append(Violation(mid, "MISSING_CRITERIA", "materiality and complexity must be High, Medium or Low"))
        elif rule is not None and exp != rating:
            s = criteria_score(r["materiality"], r["complexity"], r["uses_ai_ml"], r["regulatory_use"])
            out.append(Violation(mid, "RATING_CONTRADICTS_CRITERIA",
                                 f"rated {rating} but criteria score {s} implies {exp}"))
        if status == DECOMMISSIONED:
            ls, dd = r["last_scored_date"], r["decommissioned_date"]
            if _missing(dd):
                out.append(Violation(mid, "MISSING_DECOMMISSION_DATE", "decommissioned model has no decommission date"))
            elif not _missing(ls) and ls > dd:
                out.append(Violation(mid, "DECOMMISSIONED_STILL_SCORING",
                                     f"decommissioned {dd} but produced scores on {ls}"))
            continue
        if not active or rule is None:
            continue
        lr, nr, lm = r["last_review"], r["next_review"], r["last_monitoring"]
        if _missing(nr):
            out.append(Violation(mid, "MISSING_NEXT_REVIEW", "active model has no next review date"))
        elif nr < as_of:
            out.append(Violation(mid, "REVIEW_OVERDUE", f"next review was due {nr} (as of {as_of})"))
        if not _missing(lr) and not _missing(nr) and nr > add_months(lr, rule["review_months"]):
            out.append(Violation(mid, "REVIEW_INTERVAL_TOO_LONG",
                                 f"{rating} models need review every {rule['review_months']} months; "
                                 f"{lr} to {nr} is longer"))
        try:
            interval = int(float(r["monitoring_interval_months"]))
        except ValueError:
            interval = None
            out.append(Violation(mid, "MISSING_MONITORING_INTERVAL", "no monitoring interval recorded"))
        if interval is not None and interval > rule["monitoring_months"]:
            out.append(Violation(mid, "MONITORING_INTERVAL_TOO_LONG",
                                 f"{rating} models need monitoring every {rule['monitoring_months']} month(s); "
                                 f"inventory says {interval}"))
        if _missing(lm):
            out.append(Violation(mid, "MISSING_MONITORING", "active model has no monitoring on record"))
        else:
            allowed = min(interval or rule["monitoring_months"], rule["monitoring_months"])
            due = add_months(lm, allowed)
            if due < as_of:
                out.append(Violation(mid, "MONITORING_OVERDUE", f"last monitored {lm}; next run was due {due}"))
    return out


def violations_frame(v: list[Violation]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in v], columns=["model_id", "rule", "detail"])

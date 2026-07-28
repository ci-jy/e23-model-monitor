from datetime import date

import pandas as pd
import pytest

from e23mon.inventory import (RATING_RULES, add_months, expected_rating, load_inventory, validate,
                              violations_frame)

PLANTED = {
    ("CR-PD-002", "MONITORING_INTERVAL_TOO_LONG"),
    ("CR-LGD-003", "RATING_CONTRADICTS_CRITERIA"),
    ("PR-AUTO-004", "REVIEW_OVERDUE"),
    ("AML-TM-006", "MISSING_OWNER"),
    ("AML-CRR-007", "MONITORING_OVERDUE"),
    ("AI-GEN-008", "REVIEW_INTERVAL_TOO_LONG"),
    ("CR-BHV-011", "DECOMMISSIONED_STILL_SCORING"),
}
CLEAN = {"CR-PD-001", "PR-HOME-005", "AI-OCR-009", "FIN-ALLOC-010"}


@pytest.mark.parametrize("path", ["examples/inventory.csv", "examples/inventory.yaml"])
def test_example_inventory_flags_each_planted_violation(path):
    inv = load_inventory(path)
    v = violations_frame(validate(inv))
    found = set(zip(v.model_id, v.rule))
    assert PLANTED <= found
    assert not (set(v.model_id) & CLEAN)
    assert set(inv.risk_rating) == {"High", "Medium", "Low"}


def test_rating_rules():
    assert RATING_RULES["High"]["review_months"] < RATING_RULES["Medium"]["review_months"] < RATING_RULES["Low"]["review_months"]
    assert expected_rating("High", "Medium", "no", "yes") == "High"      # 6 + 2 + 1 = 9
    assert expected_rating("High", "Low") == "Medium"                     # 6 + 1 = 7
    assert expected_rating("Medium", "High", "yes") == "High"             # 4 + 3 + 1 = 8
    assert expected_rating("Low", "Medium") == "Low"                      # 2 + 2 = 4
    assert expected_rating("Low", "Medium", "yes") == "Medium"            # 5
    assert expected_rating("Huge", "Low") is None


def test_add_months_end_of_month():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 11, 15), 3) == date(2027, 2, 15)


def _one(**kw):
    row = dict(model_id="M1", name="m", owner="Risk", business_use="x", category="Credit", materiality="Medium",
               complexity="Medium", uses_ai_ml="no", regulatory_use="no", risk_rating="Medium", status="Active",
               last_review="2026-01-01", next_review="2027-06-01", monitoring_interval_months="3",
               last_monitoring="2026-09-01", decommissioned_date="", last_scored_date="2026-09-30")
    row.update(kw)
    return pd.DataFrame([row])


def _rules(df, tmp_path, as_of=date(2026, 10, 1)):
    p = tmp_path / "inv.csv"
    df.to_csv(p, index=False)
    return {v.rule for v in validate(load_inventory(p), as_of)}


def test_clean_model_has_no_violations(tmp_path):
    assert _rules(_one(), tmp_path) == set()


def test_each_rule_triggers(tmp_path):
    assert "MISSING_OWNER" in _rules(_one(owner=""), tmp_path)
    assert "INVALID_RATING" in _rules(_one(risk_rating="Critical"), tmp_path)
    assert "RATING_CONTRADICTS_CRITERIA" in _rules(_one(risk_rating="Low"), tmp_path)
    assert "REVIEW_OVERDUE" in _rules(_one(next_review="2026-09-30"), tmp_path)
    assert "REVIEW_INTERVAL_TOO_LONG" in _rules(_one(next_review="2028-06-01"), tmp_path)
    assert "MONITORING_INTERVAL_TOO_LONG" in _rules(_one(monitoring_interval_months="6"), tmp_path)
    assert "MONITORING_OVERDUE" in _rules(_one(last_monitoring="2026-06-01"), tmp_path)
    assert "MISSING_MONITORING" in _rules(_one(last_monitoring=""), tmp_path)
    dec = _one(status="Decommissioned", decommissioned_date="2026-03-31", last_scored_date="2026-04-02")
    assert _rules(dec, tmp_path) == {"DECOMMISSIONED_STILL_SCORING"}
    ok = _one(status="Decommissioned", decommissioned_date="2026-03-31", last_scored_date="2026-03-30",
              next_review="2020-01-01")
    assert _rules(ok, tmp_path) == set()  # retired models are not due for review
    dup = pd.concat([_one(), _one()])
    assert "DUPLICATE_ID" in _rules(dup, tmp_path)


def test_as_of_changes_overdue_status(tmp_path):
    df = _one(next_review="2026-12-01", last_monitoring="2026-09-01")
    assert _rules(df, tmp_path, date(2026, 10, 1)) == set()
    assert {"REVIEW_OVERDUE", "MONITORING_OVERDUE"} <= _rules(df, tmp_path, date(2027, 1, 15))

"""Run the false-alarm and power simulation on the UCI data and save the results.

Usage: python3 scripts/run_simulation.py [--quick]
Writes docs/results/simulation_raw.csv.gz, false_alarm.csv and power.csv.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e23mon.simulate import run, summarize_false_alarms, summarize_power  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run for a smoke test")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args()
    kw = dict(n_null=1000, n_reps=1000, n_power=200)
    if a.quick:
        kw = dict(sizes=(500, 5000), n_null=100, n_reps=100, n_power=20)
    t = time.time()
    raw = run(workers=a.workers, **kw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "simulation_raw.csv.gz", index=False)
    fa = summarize_false_alarms(raw)
    pw = summarize_power(raw)
    fa.to_csv(out / "false_alarm.csv", index=False)
    pw.to_csv(out / "power.csv", index=False)
    print(fa.to_string(index=False))
    print(pw.to_string(index=False))
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

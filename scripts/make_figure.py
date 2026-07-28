"""Plot false-alarm rates and detection power from docs/results/*.csv into
docs/false_alarm_power.png and print markdown tables for the README."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"


def main():
    fa = pd.read_csv(RES / "false_alarm.csv")
    pw = pd.read_csv(RES / "power.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    ax = axes[0]
    x = fa.slice_size
    ax.fill_between(x, fa.nominal_low, fa.nominal_high, color="0.85", label="95% binomial band of 5%")
    ax.plot(x, fa.model_unadjusted, "o-", color="tab:red", label="all metrics at p<0.05, no correction")
    ax.plot(x, fa.model_rot, "s-", color="tab:orange", label="rule-of-thumb policy (any metric)")
    ax.plot(x, fa.model_holm, "D-", color="tab:blue", label="calibrated + Holm (any metric)")
    ax.plot(x, fa.psi_calibrated, "^--", color="tab:green", label="score PSI > calibrated 5% threshold")
    ax.plot(x, fa.psi_rot_010, "v:", color="0.4", label="score PSI >= 0.10")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("slice size")
    ax.set_ylabel("false-alarm rate per model per period")
    ax.set_title(f"No drift ({int(fa.slices.iloc[0])} slices per size)")
    ax.legend(fontsize=7, loc="upper left")
    titles = {"covariate": "Covariate drift (tilt on credit limit)", "label": "Label drift (default rate x (1+d))",
              "calibration": "Calibration drift (share of goods turned bad)"}
    sizes = [1000, 10000]
    for ax, kind in zip(axes[1:], ["covariate", "label", "calibration"]):
        d = pw[pw.drift == kind]
        for n, style in zip(sizes, ["-", "--"]):
            g = d[d.slice_size == n].sort_values("magnitude")
            ax.plot(g.magnitude, g.model_holm, "D" + style, color="tab:blue", label=f"calibrated + Holm, n={n}")
            ax.plot(g.magnitude, g.model_rot, "s" + style, color="tab:orange", label=f"rule-of-thumb, n={n}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("drift magnitude d")
        ax.set_ylabel("detection power")
        ax.set_title(titles[kind])
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("UCI credit default model (gradient boosting): false alarms and detection power", fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs" / "false_alarm_power.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    print("\n| slice size | score PSI >= 0.10 | score PSI >= 0.25 | score PSI > calibrated 5% threshold "
          "| any metric, no correction | rule-of-thumb policy | calibrated + Holm |")
    print("|---|---|---|---|---|---|---|")
    for _, r in fa.iterrows():
        print(f"| {r.slice_size:,} | {pct(r.psi_rot_010)} | {pct(r.psi_rot_025)} | {pct(r.psi_calibrated)} "
              f"| {pct(r.model_unadjusted)} | {pct(r.model_rot)} | {pct(r.model_holm)} |")
    print("\n| drift | magnitude | " + " | ".join(f"n={n:,} (cal / RoT)" for n in sorted(pw.slice_size.unique())) + " |")
    print("|---|---|" + "---|" * pw.slice_size.nunique())
    for (kind, mag), g in pw.groupby(["drift", "magnitude"], sort=False):
        g = g.sort_values("slice_size")
        cells = " | ".join(f"{pct(a)} / {pct(b)}" for a, b in zip(g.model_holm, g.model_rot))
        print(f"| {kind} | {mag} | {cells} |")


if __name__ == "__main__":
    sys.exit(main())

"""
ctd_carb_model.report
=====================
PART 5 of the modelling pipeline: diagnostics and reporting.

Turns the comparison (Part 4) into the artifacts that convince a reader, all
built from OUT-OF-GROUP predictions (Part 3) so they show honest skill:

  - predicted_vs_actual_plot : the headline scatter (1:1 line, fit, R2/RMSE)
  - residual_plot            : residual vs predicted, to expose structure/bias
  - per_fold_plot            : one panel per held-out cast (no fold hidden)
  - comparison_bar_plot      : MLR vs ANN R2/RMSE side by side
  - write_report             : a markdown report tying it together
  - write_manifest           : a JSON record for reproducibility

matplotlib is imported lazily so importing this module never needs a display.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import validation


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required for plots. pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Plots (each takes a cross_validate result dict for one response+model)
# ---------------------------------------------------------------------------


def predicted_vs_actual_plot(cv: dict, model_name: str, out_path) -> Path:
    plt = _mpl()
    preds = cv["predictions"]
    a = preds["actual"].to_numpy(); p = preds["predicted"].to_numpy()
    m = validation.metrics(a, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(a, p, s=28, alpha=0.75, edgecolor="#333", linewidth=0.4, color="#2c7fb8")
    lo = float(min(a.min(), p.min())); hi = float(max(a.max(), p.max()))
    pad = (hi - lo) * 0.05 or 1.0
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
    if np.ptp(a) > 0:
        slope, intercept = np.polyfit(a, p, 1)
        xs = np.array(lims)
        ax.plot(xs, slope * xs + intercept, "-", color="#d95f0e", linewidth=1.3, label="fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Measured {cv['response']}")
    ax.set_ylabel(f"Predicted (out-of-group)")
    ax.set_title(f"{cv['response']} — {model_name}\n{cv['split']}", fontsize=10)
    txt = f"R² = {m['r2']:.3f}\nRMSE = {m['rmse']:.3g}\nbias = {m['bias']:+.3g}\nn = {m['n']}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_path)


def residual_plot(cv: dict, model_name: str, out_path) -> Path:
    plt = _mpl()
    preds = cv["predictions"]
    a = preds["actual"].to_numpy(); p = preds["predicted"].to_numpy()
    r = a - p
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(p, r, s=24, alpha=0.7, color="#2c7fb8", edgecolor="#333", linewidth=0.3)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel(f"Predicted {cv['response']}")
    ax.set_ylabel("Residual (measured − predicted)")
    ax.set_title(f"{cv['response']} — {model_name} residuals", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_path)


def per_fold_plot(cv: dict, model_name: str, out_path) -> Path:
    plt = _mpl()
    preds = cv["predictions"]
    groups = list(preds["held_out"].dropna().unique()) or ["all"]
    ncol = min(4, len(groups)); nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for i, g in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        gp = preds[preds["held_out"] == g]
        a = gp["actual"].to_numpy(); p = gp["predicted"].to_numpy()
        ax.scatter(a, p, s=22, alpha=0.8, color="#2c7fb8", edgecolor="#333", linewidth=0.3)
        if len(a):
            lo = float(min(a.min(), p.min())); hi = float(max(a.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
            mm = validation.metrics(a, p)
            ax.set_title(f"{g}\nRMSE={mm['rmse']:.3g}, n={mm['n']}", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{cv['response']} — {model_name}: per held-out cast", fontsize=10)
    fig.supxlabel("Measured", fontsize=9); fig.supylabel("Predicted", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    return _save(fig, out_path)


def comparison_bar_plot(table: pd.DataFrame, out_path) -> Path:
    plt = _mpl()
    t = table[(table.get("ok", True) == True) & np.isfinite(table.get("r2", np.nan))].copy()
    if t.empty:
        raise ValueError("no validated rows to plot.")
    # clip very negative R2 so a catastrophic model doesn't crush the scale
    t["r2_plot"] = t["r2"].clip(lower=-1.0)
    labels = [f"{r.response}\n{r.model}" for r in t.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(max(6, 1.5 * len(t)), 4))
    colors = ["#2c7fb8" if m == "MLR" else "#d95f0e" for m in t["model"]]
    axes[0].bar(labels, t["r2_plot"], color=colors); axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_title("Out-of-group R² (clipped at −1)", fontsize=10); axes[0].set_ylim(top=1.0)
    axes[0].tick_params(axis="x", labelsize=7); axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(labels, t["rmse"], color=colors); axes[1].set_title("Out-of-group RMSE", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=7); axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration: all diagnostics for a comparison result
# ---------------------------------------------------------------------------


def generate_all(result: dict, out_dir, response_status: dict | None = None) -> dict:
    """Produce every plot + the comparison table CSV for a Part-4 result."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = {"plots": [], "tables": []}

    tpath = out_dir / "model_comparison_table.csv"
    result["table"].to_csv(tpath, index=False); written["tables"].append(str(tpath))
    wpath = out_dir / "winners.csv"
    result["winners"].to_csv(wpath, index=False); written["tables"].append(str(wpath))

    # predictor selection rankings, if selection was run
    for resp, info in (result.get("selection") or {}).items():
        ranking = info.get("ranking")
        if ranking is not None:
            rp = out_dir / f"{resp}_predictor_selection.csv"
            ranking.to_csv(rp, index=False); written["tables"].append(str(rp))

    try:
        written["plots"].append(str(comparison_bar_plot(
            result["table"], out_dir / "comparison_bars.png")))
    except ValueError:
        pass

    for (response, model), cv in result["cv"].items():
        if not cv.get("ok"):
            continue
        tag = f"{response}_{model}"
        written["plots"] += [
            str(predicted_vs_actual_plot(cv, model, out_dir / f"{tag}_pred_vs_actual.png")),
            str(residual_plot(cv, model, out_dir / f"{tag}_residuals.png")),
            str(per_fold_plot(cv, model, out_dir / f"{tag}_per_fold.png")),
        ]
        cv["predictions"].to_csv(out_dir / f"{tag}_out_of_group_predictions.csv", index=False)
    return written


# ---------------------------------------------------------------------------
# Report + manifest
# ---------------------------------------------------------------------------


def write_report(result: dict, out_path, dataset_summary: str = "",
                 response_status: dict | None = None) -> Path:
    from .compare import format_comparison
    lines = [
        "# Carbonate modelling report", "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", "",
        "## Dataset", "```", dataset_summary or "(not provided)", "```", "",
        "## Model comparison", "```", format_comparison(result), "```", "",
    ]
    if response_status:
        calc = [r for r, s in response_status.items() if s == "calculated"]
        if calc:
            lines += [
                "## Note on calculated variables",
                "These responses are derived from the measured pH+TA pair, not "
                "modelled directly: " + ", ".join(calc) + ". Predicted values for "
                "them are obtained by recomputing from the winning model's "
                "predicted pH and TA via PyCO2SYS, so they stay carbonate-system-"
                "consistent. They are NOT validated against independent measurements.", "",
            ]
    lines += [
        "## How to read the plots",
        "- `*_pred_vs_actual.png`: out-of-group predicted vs measured; points on "
        "the 1:1 line = good. R²/RMSE shown are honest (leave-one-cast-out).",
        "- `*_residuals.png`: flat, centred-on-zero = unbiased.",
        "- `*_per_fold.png`: one panel per held-out cast; checks skill is uniform.",
        "- `comparison_bars.png`: MLR vs ANN side by side.", "",
        "## Caveats",
        "- Out-of-group metrics are the honest measure; ignore in-sample fit.",
        "- A negative R² means the model is worse than predicting the mean.",
        "- On a small / single-cruise dataset, treat models as exploratory and "
        "do not claim temporal or seasonal transfer.", "",
    ]
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_manifest(result: dict, out_path, inputs: dict | None = None,
                   dataset_summary: str = "") -> Path:
    table = result["table"]
    manifest = {
        "tool": "ctd_carb_model",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_split": result["split"],
        "inputs": inputs or {},
        "dataset_summary": dataset_summary,
        "comparison": table.to_dict(orient="records"),
        "winners": result["winners"].to_dict(orient="records"),
    }
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out_path

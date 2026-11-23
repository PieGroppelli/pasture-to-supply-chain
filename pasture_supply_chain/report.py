"""Charts and tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

GREEN, GREY, RED, BLUE = "#2e7d32", "#9e9e9e", "#c62828", "#1565c0"
HEALTH_COLOUR = {"Good": GREEN, "Fair": "#f9a825", "Poor": RED}


def field_maps(fields, images, results_target, path=None):
    """One row per field: true colour and NDVI for the target year."""
    n = len(fields)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n), squeeze=False)
    for i, f in enumerate(fields):
        r = results_target[i]
        rgb, ndvi = images[f["name"]]
        axes[i][0].imshow(rgb)
        axes[i][1].imshow(ndvi)
        axes[i][0].set_title(f"{f['name']} - true colour {r.year}", fontsize=10)
        axes[i][1].set_title(f"NDVI {r.year} | health: {r.health} (mean {r.mean_ndvi:.2f})",
                             fontsize=10, color=HEALTH_COLOUR[r.health])
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def ndvi_curves(results_target, results_baseline, path=None):
    n = len(results_target)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.6), squeeze=False, sharey=True)
    for i, (t, b) in enumerate(zip(results_target, results_baseline)):
        ax = axes[0][i]
        doy_t = t.daily.index.dayofyear
        doy_b = b.daily.index.dayofyear
        ax.plot(doy_b, b.daily["ndvi"], color=GREY, label=str(b.year))
        ax.plot(doy_t, t.daily["ndvi"], color=GREEN, label=str(t.year))
        ax.set_title(t.field, fontsize=10)
        ax.set_xlabel("day of year")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("NDVI (field mean)")
    axes[0][0].legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def supply_chain_chart(monthly: pd.DataFrame, target_year: int, baseline_year: int, path=None):
    """Top: monthly milk supported by the fields. Bottom: early-warning signal on grass growth."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 2]})
    x = range(len(monthly))
    w = 0.4
    ax.bar([i - w / 2 for i in x], monthly["milk_litres_baseline"] / 1000, w, color=GREY, label=str(baseline_year))
    colours = [RED if a else BLUE for a in monthly["alert"]]
    ax.bar([i + w / 2 for i in x], monthly["milk_litres_target"] / 1000, w, color=colours, label=str(target_year))
    ax.set_xticks(list(x), monthly.index)
    ax.set_ylabel("milk from pasture ('000 litres)")
    ax.set_title("Monthly milk supported by the fields (red = growth below alert threshold)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ratio = monthly.attrs.get("growth_ratio")
    alert = monthly.attrs.get("alert_date")
    if ratio is not None and len(ratio):
        dates = pd.Timestamp(f"{target_year}-01-01") + pd.to_timedelta(ratio.index - 1, unit="D")
        ax2.plot(dates, ratio.values * 100, color=BLUE)
        ax2.axhline(80, color=RED, linestyle="--", linewidth=1, label="alert threshold (80%)")
        if alert is not None:
            ax2.axvline(alert, color=RED, linewidth=1.5)
            ax2.annotate(f"first alert {alert:%d %b}", (alert, ax2.get_ylim()[1]), xytext=(5, -15),
                         textcoords="offset points", color=RED, fontsize=9)
        ax2.set_ylabel(f"30-day grass growth\n% of {baseline_year}")
        ax2.set_title("Early warning from satellite and weather data")
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

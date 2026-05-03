"""Generate all figures for the paper.

Produces:
1. Velocity-volatility scatter plot with clusters (Wikipedia)
2. Decay surface contour plot (synthetic)
3. Retrieval comparison bar chart (synthetic)
4. Lifetime distributions by cluster (Wikipedia)
5. Shape parameter comparison (Wikipedia)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12

from src.clustering.temporal_clusters import build_predicate_features, cluster_dpgmm
from src.models.bayesian.survival_model import fit_weibull, compare_distributions
from src.synthetic.generator import generate_synthetic_kg, compute_tau

FIGURE_DIR = Path("paper-neurips/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def fig1_shelf_life_by_type():
    """Shelf life distributions by predicate type, showing the core finding directly.

    Left: overlaid survival curves by predicate type (how fast each type decays).
    Right: box plot of log shelf life by predicate type with discovered clusters overlaid.
    """
    df = pd.read_parquet("data/wikipedia/temporal_kg.parquet")

    # Assign ground truth predicate type
    def get_pred_type(pred):
        if pred.startswith("category:"):
            return "Category"
        elif pred.startswith("infobox:"):
            return "Infobox"
        elif pred.startswith("lead"):
            return "Lead sentence"
        else:
            return "Section"

    df = df.copy()
    df["pred_type"] = df["predicate"].apply(get_pred_type)

    type_colors = {
        "Category":      "#1B5E20",
        "Infobox":       "#0D47A1",
        "Section":       "#E65100",
        "Lead sentence": "#B71C1C",
    }
    type_order = ["Category", "Infobox", "Section", "Lead sentence"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: Empirical survival curves ---
    ax = axes[0]
    t_range = np.linspace(0, 365 * 10, 500)  # 10 years

    for ptype in type_order:
        sub = df[df["pred_type"] == ptype]
        lifetimes = sub["lifetime_observed"].values
        n_total = len(lifetimes)
        if n_total == 0:
            continue

        # Kaplan-Meier style: fraction surviving past each time point
        survival = np.array([np.mean(lifetimes > t) for t in t_range])

        sup_rate = (~sub["is_censored"]).mean()
        median_life = np.median(lifetimes)
        ax.plot(
            t_range / 365,
            survival,
            color=type_colors[ptype],
            linewidth=2.5,
            label=f"{ptype}\n  median={median_life:.0f}d, sup.={sup_rate:.0%}",
        )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Fraction of facts still current")
    ax.set_title("(a) How fast do facts become stale?")
    ax.legend(fontsize=8, loc="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 10)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(9.5, 0.52, "50%", fontsize=8, color="gray", ha="right")

    # --- Right panel: Box plot of shelf life by type ---
    ax = axes[1]

    # Only superseded edges (observed lifetimes) for cleaner box plots
    superseded = df[~df["is_censored"]].copy()
    superseded["log_lifetime"] = np.log10(superseded["lifetime_observed"].clip(lower=0.1))

    box_data = []
    box_labels = []
    box_colors = []
    for ptype in type_order:
        sub = superseded[superseded["pred_type"] == ptype]
        if len(sub) > 0:
            box_data.append(sub["log_lifetime"].values)
            box_labels.append(ptype)
            box_colors.append(type_colors[ptype])

    bp = ax.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Add sample size annotations
    for i, (ptype, data) in enumerate(zip(box_labels, box_data)):
        ax.text(i + 1, ax.get_ylim()[0] + 0.1, f"n={len(data)}",
                ha="center", fontsize=8, color="gray")

    ax.set_ylabel("log$_{10}$(shelf life in days)")
    ax.set_title("(b) Shelf life distribution (superseded facts only)")

    # Add day-scale reference lines
    for days, label in [(1, "1 day"), (30, "1 month"), (365, "1 year"), (3650, "10 years")]:
        y = np.log10(days)
        if ax.get_ylim()[0] < y < ax.get_ylim()[1]:
            ax.axhline(y=y, color="gray", linestyle=":", alpha=0.3)
            ax.text(len(box_labels) + 0.5, y, label, fontsize=7, color="gray", va="center")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_shelf_life.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig1_shelf_life.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_shelf_life")


def fig2_decay_surface():
    """Decay surface contour plot for one cluster (synthetic)."""
    # Use the volatile measurements cluster params
    theta = np.array([3.0, -0.5, -2.0, 0.2])

    v_range = np.linspace(0.01, 1.5, 100)
    s_range = np.linspace(0.01, 1.0, 100)
    V, S = np.meshgrid(v_range, s_range)

    TAU = np.exp(theta[0] + theta[1] * V + theta[2] * S + theta[3] * V * S)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: tau surface
    ax = axes[0]
    cs = ax.contourf(V, S, np.log10(TAU), levels=20, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("log$_{10}$($\\tau$) [days]")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Volatility")
    ax.set_title("Decay Surface: $\\tau(v, \\sigma)$")

    # Right: survival at t=30 days
    ax = axes[1]
    shape = 1.0
    t = 30.0
    SURVIVAL = np.exp(-((t / TAU) ** shape))
    cs2 = ax.contourf(V, S, SURVIVAL, levels=20, cmap="RdYlGn")
    cbar2 = fig.colorbar(cs2, ax=ax)
    cbar2.set_label("S(30 days)")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Volatility")
    ax.set_title("Survival Probability at $t$ = 30 days")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_decay_surface.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig2_decay_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_decay_surface")


def fig3_retrieval_comparison():
    """Bar chart comparing retrieval methods (synthetic)."""
    # Load results
    with open("results/synthetic_experiment.json") as f:
        results = json.load(f)

    methods = [
        ("no_temporal", "No temporal"),
        ("uniform_exponential", "Uniform exp."),
        ("uniform_halflife", "Uniform half-life"),
        ("type_level", "Type-level (L1)"),
        ("type_cohort", "Type+Cohort (L1+2)"),
        ("full_hierarchical", "Full hier. (L1+2+3)"),
    ]

    ndcg5 = [results["retrieval"][m]["ndcg_5"] for m, _ in methods]
    ndcg10 = [results["retrieval"][m]["ndcg_10"] for m, _ in methods]
    labels = [label for _, label in methods]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, ndcg5, width, label="NDCG@5", color="#2196F3")
    bars2 = ax.bar(x + width/2, ndcg10, width, label="NDCG@10", color="#4CAF50")

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance (Synthetic KG, 200 queries)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 0.35)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_retrieval.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig3_retrieval.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_retrieval")


def fig4_lifetime_distributions():
    """Lifetime distributions by predicate type (Wikipedia)."""
    df = pd.read_parquet("data/wikipedia/temporal_kg.parquet")

    # Only superseded edges (observed lifetimes)
    superseded = df[~df["is_censored"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    pred_types = [
        ("category:", "Categories", axes[0, 0]),
        ("infobox:", "Infobox Fields", axes[0, 1]),
        ("section:", "Section Content", axes[1, 0]),
        ("lead", "Lead Sentences", axes[1, 1]),
    ]

    for prefix, title, ax in pred_types:
        mask = superseded["predicate"].str.startswith(prefix)
        lifetimes = superseded[mask]["lifetime_observed"].values
        lifetimes = lifetimes[lifetimes > 0]

        if len(lifetimes) > 0:
            ax.hist(np.log10(lifetimes + 1), bins=30, color="#2196F3",
                    alpha=0.7, edgecolor="white", linewidth=0.5)
            median = np.median(lifetimes)
            ax.axvline(np.log10(median + 1), color="red", linestyle="--",
                       label=f"Median: {median:.0f}d")
            ax.legend(fontsize=8)

        ax.set_title(title)
        ax.set_xlabel("log$_{10}$(lifetime + 1) [days]")
        ax.set_ylabel("Count")

    fig.suptitle("Lifetime Distributions by Knowledge Type (Wikipedia)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_lifetimes.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig4_lifetimes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_lifetimes")


def fig5_uniform_decay_failure():
    """Illustrate uniform decay failure: same curve applied to permanent and volatile facts."""
    t = np.linspace(0, 365 * 5, 500)  # 5 years

    # Uniform decay (half-life = 1 year)
    tau_uniform = 365.0
    s_uniform = np.exp(-(t / tau_uniform))

    # Heterogeneous: permanent fact
    tau_perm = 3847.0
    kappa_perm = 0.7
    s_perm = np.exp(-((t / tau_perm) ** kappa_perm))

    # Heterogeneous: volatile measurement
    tau_vol = 4.3
    kappa_vol = 0.8
    s_vol = np.exp(-((t / tau_vol) ** kappa_vol))

    # Heterogeneous: current state
    tau_curr = 98.0
    kappa_curr = 0.9
    s_curr = np.exp(-((t / tau_curr) ** kappa_curr))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: uniform decay applied to all
    ax = axes[0]
    ax.plot(t / 365, s_uniform, 'k-', linewidth=2, label="All edges (uniform)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Temporal weight")
    ax.set_title("Uniform Decay (status quo)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.annotate("Permanent facts\nsuppressed", xy=(3, 0.05), fontsize=8,
                color="red", ha="center")
    ax.annotate("Stale values\nnot suppressed\nfast enough", xy=(0.3, 0.75), fontsize=8,
                color="red", ha="center")

    # Right: heterogeneous decay
    ax = axes[1]
    ax.plot(t / 365, s_perm, '-', linewidth=2, color="#4CAF50",
            label=f"Permanent ($\\tau$={tau_perm:.0f}d, $\\kappa$={kappa_perm})")
    ax.plot(t / 365, s_curr, '-', linewidth=2, color="#FF9800",
            label=f"Current state ($\\tau$={tau_curr:.0f}d, $\\kappa$={kappa_curr})")
    ax.plot(t / 365, s_vol, '-', linewidth=2, color="#F44336",
            label=f"Volatile ($\\tau$={tau_vol:.1f}d, $\\kappa$={kappa_vol})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Temporal weight")
    ax.set_title("Hierarchical Decay (ours)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig5_uniform_vs_hierarchical.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig5_uniform_vs_hierarchical.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_uniform_vs_hierarchical")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("Generating figures...")

    fig1_shelf_life_by_type()
    fig2_decay_surface()
    fig3_retrieval_comparison()
    fig4_lifetime_distributions()
    fig5_uniform_decay_failure()

    print(f"\nAll figures saved to {FIGURE_DIR}/")

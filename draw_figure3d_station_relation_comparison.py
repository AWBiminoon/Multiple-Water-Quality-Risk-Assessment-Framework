from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "graphsage_position_feature_ablation_outputs"
    / "station_metric_panels"
    / "station_metrics_four_graphsage_configs.csv"
)
OUT = ROOT / "figure3_station_relation_comparison_outputs"

CONFIG_MAP = {
    "No position + no feature": "Neither",
    "No position + feature": "Feature only",
    "Position + feature": "Position + feature",
}
ORDER = ["Neither", "Feature only", "Position + feature"]
COLORS = {
    "Neither": "#A7B1BA",
    "Feature only": "#9DDFE4",
    "Position + feature": "#4D77A7",
}
MARKERS = {"Neither": "^", "Feature only": "D", "Position + feature": "o"}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_table() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    selected = raw.loc[raw["Model"].isin(CONFIG_MAP), ["Model", "station_id", "Macro-F1"]].copy()
    selected["Configuration"] = selected["Model"].map(CONFIG_MAP)
    wide = selected.pivot(index="station_id", columns="Configuration", values="Macro-F1")
    wide = wide.loc[:, ORDER]
    wide = wide.sort_values("Position + feature", ascending=True)
    return wide


def draw(wide: pd.DataFrame) -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    wide.reset_index().to_csv(
        OUT / "Figure3d_station_relation_comparison_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor="white")
    y = np.arange(len(wide))

    for row_index, (_, row) in enumerate(wide.iterrows()):
        ax.plot(
            [row[name] for name in ORDER],
            [row_index] * len(ORDER),
            color="#C8D2DC",
            linewidth=0.8,
            zorder=1,
        )

    for name in ORDER:
        ax.scatter(
            wide[name],
            y,
            s=31,
            marker=MARKERS[name],
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
            label=name,
        )

    ax.set_yticks(y, wide.index)
    ax.set_xlabel("Station Macro-F1")
    ax.set_ylabel("Monitoring station")
    ax.set_xlim(0.25, 1.015)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    ax.grid(axis="x", color="#D7E0E8", linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=0.8, length=3.2)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.13, top=0.90)

    stem = OUT / "Figure3d_station_relation_comparison"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    draw(load_table())

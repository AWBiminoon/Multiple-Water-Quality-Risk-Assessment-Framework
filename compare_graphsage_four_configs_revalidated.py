from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import recompute_fig4_ci_cwqi as shared
import revalidate_graphsage_feature_graph as feature_validation


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "graphsage_four_config_revalidated_outputs"
OUT.mkdir(exist_ok=True)

BLUE = "#5E84B0"
BLUE_LIGHT = "#B5CADC"
GREEN = "#AECFBC"
GREEN_LIGHT = "#DCEBE3"
TEXT = "#26313D"
GRID = "#D9E2EA"

CONFIGS = [
    ("P-F-", "No position / no feature"),
    ("P+F-", "Position only"),
    ("P-F+", "Feature only"),
    ("P+F+", "Position + feature"),
]
COLORS = {
    "P-F-": "#C9D3DC",
    "P+F-": BLUE_LIGHT,
    "P-F+": GREEN,
    "P+F+": BLUE,
}


def normalized_dynamic(raw: np.ndarray) -> np.ndarray:
    return np.stack([shared.normalize_adjacency(matrix) for matrix in raw])


def build_four_graphs(data: dict) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    n_times = len(data["time_ids"])
    n_stations = len(data["station_ids"])
    identity = np.eye(n_stations, dtype=np.float32)
    identity_dynamic = np.tile(identity, (n_times, 1, 1))

    static_position_raw = data["graph_variants"]["raw"]["position_no_feature"]
    position_relation = np.maximum(static_position_raw - identity, 0.0)
    position_raw = np.tile(
        identity + position_relation,
        (n_times, 1, 1),
    )

    group_values = feature_validation.fit_group_tensor(data)
    feature_relations = []
    for time_values in group_values:
        similarity = feature_validation.rbf_similarity(time_values)
        relation = feature_validation.symmetric_knn(
            similarity,
            k=1,
            mutual=True,
        )
        feature_relations.append(relation)
    feature_relations = np.stack(feature_relations).astype(np.float32)

    feature_alpha = 0.50
    feature_raw = identity_dynamic + feature_alpha * feature_relations
    fused_raw = (
        identity_dynamic
        + np.tile(position_relation, (n_times, 1, 1))
        + feature_alpha * feature_relations
    )

    graphs = {
        "P-F-": normalized_dynamic(identity_dynamic),
        "P+F-": normalized_dynamic(position_raw),
        "P-F+": normalized_dynamic(feature_raw),
        "P+F+": normalized_dynamic(fused_raw),
    }

    diagnostics = []
    for code, label in CONFIGS:
        adjacency = graphs[code]
        edge_count = []
        for matrix in adjacency:
            relation = matrix > 0
            np.fill_diagonal(relation, False)
            edge_count.append(np.triu(relation, 1).sum())
        diagnostics.append(
            {
                "Configuration": code,
                "Description": label,
                "Mean undirected edges per month": float(np.mean(edge_count)),
                "Validation same-grade edge rate": (
                    feature_validation.edge_agreement(
                        adjacency, data["y"], data["val_t"]
                    )
                ),
                "Test same-grade edge rate": (
                    feature_validation.edge_agreement(
                        adjacency, data["y"], data["test_t"]
                    )
                ),
            }
        )
    return graphs, pd.DataFrame(diagnostics)


def run_comparison(
    data: dict,
    graphs: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_rows = []
    for code, label in CONFIGS:
        for seed in feature_validation.FINAL_SEEDS:
            print(f"Training {code}, seed={seed}", flush=True)
            model, record = feature_validation.train_model(
                data,
                graphs[code],
                seed,
                max_epochs=700,
                patience=75,
            )
            metrics, prediction = feature_validation.evaluate_split(
                model,
                data,
                graphs[code],
                data["test_t"],
            )
            rows.append(
                {
                    "Configuration": code,
                    "Description": label,
                    "Position edges": code[1] == "+",
                    "Feature edges": code[3] == "+",
                    "Seed": seed,
                    "Best epoch": record["best_epoch"],
                    "Validation Macro-F1": record["validation_macro_f1"],
                    "Accuracy": metrics["Accuracy"],
                    "Macro-F1": metrics["Macro-F1"],
                    "Within-one-grade accuracy": metrics[
                        "Within-one-grade accuracy"
                    ],
                }
            )
            truth = data["y"][data["test_t"]]
            for time_index, global_time_index in enumerate(data["test_t"]):
                for station_index, station_id in enumerate(data["station_ids"]):
                    prediction_rows.append(
                        {
                            "Configuration": code,
                            "Seed": seed,
                            "Time": data["time_ids"][global_time_index],
                            "Station": station_id,
                            "Truth": int(truth[time_index, station_index] + 1),
                            "Prediction": int(
                                prediction[time_index, station_index] + 1
                            ),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Macro-F1", "Within-one-grade accuracy"]
    summary = (
        results.groupby(["Configuration", "Description"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Configuration",
        "Description",
        "Accuracy mean",
        "Accuracy SD",
        "Macro-F1 mean",
        "Macro-F1 SD",
        "Within-one-grade accuracy mean",
        "Within-one-grade accuracy SD",
    ]
    order = {code: index for index, (code, _) in enumerate(CONFIGS)}
    summary["_order"] = summary["Configuration"].map(order)
    return summary.sort_values("_order").drop(columns="_order")


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def draw_figure(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    set_style()
    rng = np.random.default_rng(20260730)
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.18, 1.0],
        hspace=0.48,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    sub = grid[1, 0].subgridspec(1, 2, wspace=0.34)
    ax_interaction = fig.add_subplot(sub[0, 0])
    ax_delta = fig.add_subplot(sub[0, 1])

    metrics = [
        ("Accuracy", "Accuracy"),
        ("Macro-F1", "Macro-F1"),
        ("Within-one-grade accuracy", "±1 accuracy"),
    ]
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.27, 0.27, len(CONFIGS))
    width = 0.16
    for offset, (code, label) in zip(offsets, CONFIGS):
        subset = results[results["Configuration"] == code]
        means = np.array([subset[column].mean() for column, _ in metrics])
        stds = np.array([subset[column].std(ddof=1) for column, _ in metrics])
        ax_top.bar(
            x + offset,
            means,
            width=width,
            color=COLORS[code],
            edgecolor="white",
            linewidth=0.5,
            label=f"{code}  {label}",
            zorder=2,
        )
        ax_top.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor=TEXT,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=3,
        )
        for metric_index, (column, _) in enumerate(metrics):
            jitter = rng.normal(0, 0.010, len(subset))
            ax_top.scatter(
                np.full(len(subset), x[metric_index] + offset) + jitter,
                subset[column],
                s=10,
                facecolor=COLORS[code],
                edgecolor=TEXT,
                linewidth=0.35,
                alpha=0.88,
                zorder=4,
            )
    ax_top.set_xticks(x, [label for _, label in metrics])
    ax_top.set_ylabel("Test performance")
    ax_top.set_ylim(0.60, 1.015)
    ax_top.set_yticks(np.arange(0.60, 1.01, 0.10))
    ax_top.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax_top.legend(
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=7.2,
        handlelength=1.4,
        columnspacing=1.4,
    )

    lookup = summary.set_index("Configuration")
    for feature_present, color, marker, label in (
        (False, "#8796A5", "s", "Feature absent"),
        (True, BLUE, "o", "Feature present"),
    ):
        codes = ["P-F-", "P+F-"] if not feature_present else ["P-F+", "P+F+"]
        values = lookup.loc[codes, "Macro-F1 mean"].to_numpy()
        errors = lookup.loc[codes, "Macro-F1 SD"].to_numpy()
        ax_interaction.plot(
            [0, 1],
            values,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
        ax_interaction.errorbar(
            [0, 1],
            values,
            yerr=errors,
            fmt="none",
            ecolor=color,
            elinewidth=0.75,
            capsize=2,
            zorder=2,
        )
    ax_interaction.set_xticks([0, 1], ["Position absent", "Position present"])
    ax_interaction.set_ylabel("Macro-F1")
    ax_interaction.set_ylim(0.60, 0.82)
    ax_interaction.grid(axis="y", color=GRID, linewidth=0.55)
    ax_interaction.legend(loc="lower left", fontsize=7.2)

    baseline = float(lookup.loc["P-F-", "Macro-F1 mean"])
    codes = [code for code, _ in CONFIGS]
    changes = (
        lookup.loc[codes, "Macro-F1 mean"].to_numpy() - baseline
    )
    y = np.arange(len(codes))
    ax_delta.axvline(0, color=TEXT, linewidth=0.8)
    ax_delta.barh(
        y,
        changes,
        color=[COLORS[code] for code in codes],
        height=0.58,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_delta.set_yticks(y, codes)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("Δ Macro-F1 vs P-F-")
    ax_delta.grid(axis="x", color=GRID, linewidth=0.55)
    limit = max(0.08, float(np.max(np.abs(changes))) + 0.025)
    ax_delta.set_xlim(-limit, 0.055)
    for index, change in enumerate(changes):
        label_x = 0.006 if change >= 0 else change + 0.008
        ax_delta.text(
            label_x,
            index,
            f"{change:+.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    for label, axis in zip("abc", [ax_top, ax_interaction, ax_delta]):
        axis.text(
            -0.10,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.92)
    stem = OUT / "GraphSAGE_four_config_revalidated"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def main() -> None:
    data = shared.load_data()
    graphs, diagnostics = build_four_graphs(data)
    results, predictions = run_comparison(data, graphs)
    summary = summarize(results)

    results.to_csv(
        OUT / "graphsage_four_config_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT / "graphsage_four_config_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUT / "graphsage_four_config_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        OUT / "graphsage_four_config_graph_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    draw_figure(results, summary)

    print("\nFour-configuration summary")
    print(summary.to_string(index=False))
    print("\nGraph diagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import compare_reduced_indicator_model_families as families
import evaluate_reduced_indicator_graph_prediction as reduced
import recompute_fig4_ci_cwqi as shared


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fig4_final_model_explainability_outputs"
MODEL_DIR = OUT / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEEDS = shared.SEEDS
N_PERMUTATION_BOOTSTRAPS = 200
N_INTERACTION_BOOTSTRAPS = 2000
IG_STEPS = 32
BLOCK_MONTHS = 6

BLUE = "#5E84B0"
BLUE_DARK = "#365F88"
BLUE_LIGHT = "#B5CADC"
GREEN = "#AECFBC"
GREEN_DARK = "#729E88"
CYAN = "#9DDFE4"
PINK = "#F5CED0"
PALE = "#E0E9F0"
TEXT = "#26313D"
GRID = "#D9E2EA"
DIVERGING = LinearSegmentedColormap.from_list("blue_white_pink", [BLUE, "#FFFFFF", PINK])

DISPLAY = {
    "总氮": "TN",
    "生化需氧量": r"BOD$_5$",
    "总磷": "TP",
    "氨氮": r"NH$_4^+$-N",
    "电导率": "Conductivity",
    "阴离子表面活性剂": "Surfactants",
    "浊度": "Turbidity",
    "化学需氧量": "COD",
    "pH": "pH",
    "粪大肠菌群": "Fecal coliforms",
    "砷": "As",
    "透明度": "Transparency",
}

GROUP_COLORS = {
    "Organic-oxygen": BLUE_DARK,
    "Nutrients": GREEN_DARK,
    "Toxic-metal": CYAN,
    "Microbial": "#8D9ECF",
    "Oxygen-acid-base": "#93A7BC",
    "Other": "#A7B1BA",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def load_final_data() -> tuple[dict, list[int], pd.DataFrame, np.ndarray]:
    data = shared.load_data()
    feature_sets, audit = reduced.select_features(data)
    indices = feature_sets[12]
    n_times = len(data["time_ids"])
    identity = np.eye(len(data["station_ids"]), dtype=np.float32)
    adjacency = np.tile(identity, (n_times, 1, 1))
    selected_audit = audit[audit["Feature"].isin([data["features"][i] for i in indices])].copy()
    selected_audit["Selected index"] = selected_audit["Feature"].map(
        {data["features"][index]: position for position, index in enumerate(indices)}
    )
    selected_audit = selected_audit.sort_values("Selected index")
    return data, indices, selected_audit, adjacency


def load_or_train_models(data: dict, x: np.ndarray, adjacency: np.ndarray) -> tuple[list, pd.DataFrame]:
    models = []
    records = []
    for seed in SEEDS:
        path = MODEL_DIR / f"gat_no_edge_seed_{seed}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = families.graph_models.make_model("GAT", x.shape[-1], families.GAT_CONFIG)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            record = payload["record"]
        else:
            print(f"Training final GAT-no-edge seed={seed}", flush=True)
            model, record = families.train_graph_model(
                "GAT", families.GAT_CONFIG, x, data["y"], data["train_t"],
                data["val_t"], adjacency, seed,
            )
            torch.save({"state_dict": model.state_dict(), "record": record}, path)
        models.append(model)
        records.append({"Seed": seed, **record})
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "model_training_audit.csv", index=False, encoding="utf-8-sig")
    return models, frame


def predict_model(model, x: np.ndarray, adjacency: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            logits = model(
                torch.tensor(x[start:stop], dtype=torch.float32),
                torch.tensor(adjacency[start:stop], dtype=torch.float32),
            )
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probability = np.concatenate(probabilities, axis=0)
    return probability.argmax(-1), probability[..., 3:].sum(-1)


def predict_flat_high(model, values: np.ndarray, chunk: int = 1024) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk):
            stop = min(start + chunk, len(values))
            batch = torch.tensor(values[start:stop, None, :], dtype=torch.float32)
            adjacency = torch.ones((len(batch), 1, 1), dtype=torch.float32)
            probabilities = torch.softmax(model(batch, adjacency), dim=-1)[..., 3:].sum(-1)
            outputs.append(probabilities[:, 0].cpu().numpy())
    return np.concatenate(outputs)


def integrated_gradients_high(model, x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    tx = torch.tensor(x, dtype=torch.float32)
    graph = torch.tensor(adjacency, dtype=torch.float32)
    baseline = torch.zeros_like(tx)
    accumulated = torch.zeros_like(tx)
    model.eval()
    for alpha in torch.linspace(1.0 / IG_STEPS, 1.0, IG_STEPS):
        interpolated = (baseline + alpha * (tx - baseline)).detach().requires_grad_(True)
        logits = model(interpolated, graph)
        probability = torch.softmax(logits, dim=-1)[..., 3:].sum()
        gradient = torch.autograd.grad(probability, interpolated, retain_graph=False)[0]
        accumulated += gradient.detach()
    return ((tx - baseline) * accumulated / IG_STEPS).cpu().numpy()


def permutation_importance(
    models: list,
    x_test: np.ndarray,
    y_test: np.ndarray,
    adjacency_test: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260801)
    n_time = len(x_test)
    bootstrap_indices = np.stack(
        [moving_block_indices(n_time, BLOCK_MONTHS, rng) for _ in range(N_PERMUTATION_BOOTSTRAPS)]
    )
    replicate_rows = []
    for model_index, (seed, model) in enumerate(zip(SEEDS, models)):
        baseline_prediction, _ = predict_model(model, x_test, adjacency_test)
        baseline_scores = np.asarray(
            [
                f1_score(
                    y_test[index].reshape(-1), baseline_prediction[index].reshape(-1),
                    average="macro", zero_division=0,
                )
                for index in bootstrap_indices
            ]
        )
        for feature_index, feature in enumerate(feature_names):
            batches = []
            for index in bootstrap_indices:
                values = x_test[index].copy()
                shuffled = values[..., feature_index].reshape(-1).copy()
                rng.shuffle(shuffled)
                values[..., feature_index] = shuffled.reshape(values.shape[:2])
                batches.append(values)
            permuted_x = np.concatenate(batches, axis=0)
            permuted_adj = np.tile(adjacency_test[:1], (len(permuted_x), 1, 1))
            prediction, _ = predict_model(model, permuted_x, permuted_adj)
            prediction = prediction.reshape(N_PERMUTATION_BOOTSTRAPS, n_time, x_test.shape[1])
            permuted_scores = np.asarray(
                [
                    f1_score(
                        y_test[index].reshape(-1), prediction[replicate].reshape(-1),
                        average="macro", zero_division=0,
                    )
                    for replicate, index in enumerate(bootstrap_indices)
                ]
            )
            for replicate, drop in enumerate(baseline_scores - permuted_scores, start=1):
                replicate_rows.append(
                    {
                        "Seed": seed,
                        "Bootstrap replicate": replicate,
                        "Feature": feature,
                        "Macro-F1 decrease": float(drop),
                    }
                )
    replicates = pd.DataFrame(replicate_rows)
    summary = (
        replicates.groupby("Feature")["Macro-F1 decrease"]
        .agg(
            Mean="mean",
            **{
                "CI low": lambda values: values.quantile(0.025),
                "CI high": lambda values: values.quantile(0.975),
            },
        )
        .reset_index()
        .sort_values("Mean", ascending=False)
    )
    return summary, replicates


def attribution_tables(
    models: list,
    x_test: np.ndarray,
    y_test: np.ndarray,
    adjacency_test: np.ndarray,
    feature_names: list[str],
    station_ids: list[str],
    time_ids: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    attributions = []
    for seed, model in zip(SEEDS, models):
        print(f"Integrated gradients seed={seed}", flush=True)
        attributions.append(integrated_gradients_high(model, x_test, adjacency_test))
    mean_attribution = np.mean(attributions, axis=0)
    rows = []
    for time_index, time_id in enumerate(time_ids):
        for station_index, station in enumerate(station_ids):
            for feature_index, feature in enumerate(feature_names):
                rows.append(
                    {
                        "time_id": time_id,
                        "station_id": station,
                        "CI-CWQI grade": int(y_test[time_index, station_index] + 1),
                        "Feature": feature,
                        "Standardized value": float(x_test[time_index, station_index, feature_index]),
                        "High-risk attribution": float(
                            mean_attribution[time_index, station_index, feature_index]
                        ),
                    }
                )
    return mean_attribution, pd.DataFrame(rows)


def ale_curves(models: list, flat_x: np.ndarray, top_features: list[str], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature in top_features:
        feature_index = feature_names.index(feature)
        edges = np.unique(np.quantile(flat_x[:, feature_index], np.linspace(0, 1, 9)))
        if len(edges) < 4:
            edges = np.linspace(flat_x[:, feature_index].min(), flat_x[:, feature_index].max(), 9)
        bins = np.clip(np.digitize(flat_x[:, feature_index], edges[1:-1]), 0, len(edges) - 2)
        counts = np.asarray([np.sum(bins == index) for index in range(len(edges) - 1)])
        centers = (edges[:-1] + edges[1:]) / 2
        for seed, model in zip(SEEDS, models):
            local_effects = []
            for index in range(len(edges) - 1):
                selected = np.flatnonzero(bins == index)
                if len(selected) == 0:
                    local_effects.append(0.0)
                    continue
                low = flat_x[selected].copy()
                high = flat_x[selected].copy()
                low[:, feature_index] = edges[index]
                high[:, feature_index] = edges[index + 1]
                local_effects.append(float(np.mean(predict_flat_high(model, high) - predict_flat_high(model, low))))
            curve = np.cumsum(local_effects)
            curve -= np.average(curve, weights=np.maximum(counts, 1))
            for center, value, count in zip(centers, curve, counts):
                rows.append(
                    {
                        "Feature": feature,
                        "Seed": seed,
                        "Standardized value": float(center),
                        "ALE high-risk probability": float(value),
                        "n": int(count),
                    }
                )
    return pd.DataFrame(rows)


def interaction_effects(
    models: list,
    x_test: np.ndarray,
    top_features: list[str],
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat_x = x_test.reshape(-1, x_test.shape[-1])
    n_time, n_station = x_test.shape[:2]
    pair_rows = []
    replicate_rows = []
    rng = np.random.default_rng(20260802)
    bootstrap_indices = np.stack(
        [moving_block_indices(n_time, BLOCK_MONTHS, rng) for _ in range(N_INTERACTION_BOOTSTRAPS)]
    )
    for first_index in range(len(top_features)):
        for second_index in range(first_index + 1, len(top_features)):
            first, second = top_features[first_index], top_features[second_index]
            i, j = feature_names.index(first), feature_names.index(second)
            q50_i, q80_i = np.quantile(flat_x[:, i], [0.50, 0.80])
            q50_j, q80_j = np.quantile(flat_x[:, j], [0.50, 0.80])
            model_effects = []
            for model in models:
                variants = []
                for value_i, value_j in ((q80_i, q80_j), (q80_i, q50_j), (q50_i, q80_j), (q50_i, q50_j)):
                    values = flat_x.copy()
                    values[:, i] = value_i
                    values[:, j] = value_j
                    variants.append(predict_flat_high(model, values))
                model_effects.append(variants[0] - variants[1] - variants[2] + variants[3])
            effect = np.mean(model_effects, axis=0).reshape(n_time, n_station) * 100
            point = float(effect.mean())
            bootstrap = np.asarray([effect[index].mean() for index in bootstrap_indices])
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            pair = f"{DISPLAY[first]} × {DISPLAY[second]}"
            pair_rows.append(
                {
                    "Indicator pair": pair,
                    "Feature 1": first,
                    "Feature 2": second,
                    "Interaction excess (percentage points)": point,
                    "CI low": float(low),
                    "CI high": float(high),
                }
            )
            for replicate, value in enumerate(bootstrap, start=1):
                replicate_rows.append(
                    {
                        "Indicator pair": pair,
                        "Bootstrap replicate": replicate,
                        "Interaction excess (percentage points)": float(value),
                    }
                )
    summary = pd.DataFrame(pair_rows)
    summary["Absolute effect"] = summary["Interaction excess (percentage points)"].abs()
    summary = summary.sort_values("Absolute effect", ascending=False).drop(columns="Absolute effect")
    return summary, pd.DataFrame(replicate_rows)


def add_panel_label(axis, label: str, x: float = -0.18, y: float = 1.04) -> None:
    axis.text(x, y, label, transform=axis.transAxes, fontsize=9, fontweight="bold")


def draw_panel_a(ax, importance: pd.DataFrame, audit: pd.DataFrame) -> None:
    ordered = importance.sort_values("Mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    lower = ordered["Mean"] - ordered["CI low"]
    upper = ordered["CI high"] - ordered["Mean"]
    ax.errorbar(
        ordered["Mean"], y, xerr=np.vstack([lower, upper]), fmt="none",
        ecolor="#8493A1", elinewidth=0.65, capsize=1.5,
    )
    ax.scatter(ordered["Mean"], y, color=BLUE, s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=0.65)
    ax.set_yticks(y, [DISPLAY[name] for name in ordered["Feature"]])
    ax.set_xlabel("Decrease in test Macro-F1")
    ax.set_ylabel("Water-quality indicator")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    add_panel_label(ax, "a")


def draw_panel_b(fig, ax, attribution: pd.DataFrame, importance: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260803)
    order = importance.sort_values("Mean", ascending=False)["Feature"].tolist()
    for position, feature in enumerate(order):
        values = attribution[attribution["Feature"] == feature]
        y = position + rng.normal(0, 0.085, len(values))
        scatter = ax.scatter(
            values["High-risk attribution"], y,
            c=np.clip(values["Standardized value"], -2, 2), cmap=DIVERGING,
            vmin=-2, vmax=2, s=4.2, alpha=0.62, linewidth=0,
        )
    ax.axvline(0, color=TEXT, linewidth=0.65)
    ax.set_yticks(np.arange(len(order)), [DISPLAY[name] for name in order])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to Grade IV–V probability")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.040, pad=0.022)
    colorbar.set_ticks([-2, 2])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Standardized value", fontsize=5.5)
    colorbar.ax.tick_params(labelsize=4.9, width=0.5, length=1.8)
    add_panel_label(ax, "b")


def draw_panel_c(fig, ax, grade_table: pd.DataFrame, order: list[str]) -> None:
    raw = grade_table.set_index("Feature").loc[order, [1, 2, 3, 4, 5]].to_numpy(float)
    centered = raw - raw.mean(axis=1, keepdims=True)
    scale = np.maximum(np.max(np.abs(centered), axis=1, keepdims=True), 1e-8)
    matrix = centered / scale
    edges_x = np.arange(6) - 0.5
    edges_y = np.arange(len(order) + 1) - 0.5
    mesh = ax.pcolormesh(
        edges_x, edges_y, matrix, cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        shading="flat", edgecolors="white", linewidth=0.45,
    )
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(5), ["I", "II", "III", "IV", "V"])
    ax.set_yticks(range(len(order)), [DISPLAY[name] for name in order])
    ax.set_xlabel("CI-CWQI grade")
    ax.set_ylabel("Water-quality indicator")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.043, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.2, width=0.5, length=2)
    colorbar.set_label("Relative attribution", fontsize=5.8)
    add_panel_label(ax, "c")


def draw_panel_d(ax, ale: pd.DataFrame, top_features: list[str]) -> None:
    colors = [BLUE_DARK, GREEN_DARK, "#8D9ECF", BLUE, CYAN, "#93A7BC"]
    for feature, color in zip(top_features, colors):
        subset = ale[ale["Feature"] == feature]
        grouped = subset.groupby("Standardized value")["ALE high-risk probability"].agg(["mean", "std"]).reset_index()
        percentiles = (np.arange(len(grouped)) + 0.5) / len(grouped) * 100
        ax.plot(percentiles, grouped["mean"], color=color, marker="o", markersize=2.2, label=DISPLAY[feature])
        ax.fill_between(
            percentiles, grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"], color=color, alpha=0.13, linewidth=0,
        )
    ax.axhline(0, color="#8E9AA5", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Indicator percentile")
    ax.set_ylabel("ALE on Grade IV–V probability")
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(ncol=2, loc="upper left", fontsize=4.8, handlelength=1.5, columnspacing=0.8)
    add_panel_label(ax, "d")


def draw_panel_e(ax, interactions: pd.DataFrame, importance: pd.DataFrame) -> None:
    shown = interactions.head(8).sort_values("Interaction excess (percentage points)").reset_index(drop=True)
    y = np.arange(len(shown))
    point = shown["Interaction excess (percentage points)"].to_numpy()
    lower = point - shown["CI low"].to_numpy()
    upper = shown["CI high"].to_numpy() - point
    colors = [GREEN_DARK if value >= 0 else BLUE for value in point]
    ax.errorbar(
        point, y, xerr=np.vstack([lower, upper]), fmt="none",
        ecolor="#7E8D9B", elinewidth=0.65, capsize=1.4,
    )
    ax.scatter(point, y, c=colors, s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=0.65)
    ax.set_yticks(y, shown["Indicator pair"])
    ax.set_xlabel("High–high interaction excess (percentage points)")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    add_panel_label(ax, "e")


def draw_panel_f(fig, ax, station_table: pd.DataFrame, top_features: list[str], station_order: list[str]) -> None:
    table = station_table.pivot(index="station_id", columns="Feature", values="Mean attribution").loc[
        station_order, top_features
    ]
    values = table.to_numpy(float)
    bound = max(abs(values.min()), abs(values.max()), 1e-5)
    sizes = 8 + 70 * np.abs(values) / bound
    scatter = None
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for row in range(len(station_order)):
        scatter = ax.scatter(
            np.arange(len(top_features)), np.full(len(top_features), row),
            c=values[row], s=sizes[row], cmap=DIVERGING,
            norm=norm,
            edgecolor="white", linewidth=0.35,
        )
    ax.set_xticks(range(len(top_features)), [DISPLAY[name] for name in top_features], rotation=35, ha="right")
    ax.set_yticks(range(len(station_order)), station_order)
    ax.set_xlim(-0.55, len(top_features) - 0.45)
    ax.set_ylim(len(station_order) - 0.5, -0.5)
    ax.set_xlabel("Dominant indicator")
    ax.set_ylabel("Monitoring station")
    ax.grid(color=GRID, linewidth=0.35)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.040, pad=0.020)
    colorbar.set_label("Mean attribution", fontsize=5.4)
    colorbar.ax.tick_params(labelsize=4.8, width=0.5, length=1.8)
    add_panel_label(ax, "f")


def draw_full_figure(
    importance: pd.DataFrame,
    audit: pd.DataFrame,
    attribution: pd.DataFrame,
    grade_table: pd.DataFrame,
    ale: pd.DataFrame,
    interactions: pd.DataFrame,
    station_table: pd.DataFrame,
    station_order: list[str],
) -> None:
    setup_style()
    order = importance.sort_values("Mean", ascending=False)["Feature"].tolist()
    top_features = order[:6]
    fig = plt.figure(figsize=(7.2, 5.4), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.09, top=0.975, wspace=0.72, hspace=0.40)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])
    draw_panel_a(ax_a, importance, audit)
    draw_panel_b(fig, ax_b, attribution, importance)
    draw_panel_c(fig, ax_c, grade_table, order)
    draw_panel_d(ax_d, ale, top_features)
    draw_panel_e(ax_e, interactions, importance)
    draw_panel_f(fig, ax_f, station_table, top_features, station_order)
    save_figure(fig, OUT / "Fig4_final_model_explainability_3x2")
    plt.close(fig)


def draw_individual_panels(
    importance: pd.DataFrame,
    audit: pd.DataFrame,
    attribution: pd.DataFrame,
    grade_table: pd.DataFrame,
    ale: pd.DataFrame,
    interactions: pd.DataFrame,
    station_table: pd.DataFrame,
    station_order: list[str],
) -> None:
    setup_style()
    order = importance.sort_values("Mean", ascending=False)["Feature"].tolist()
    top_features = order[:6]

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_a(ax, importance, audit)
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.18, top=0.95)
    save_figure(fig, OUT / "Fig4a_global_permutation_importance")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_b(fig, ax, attribution, importance)
    fig.subplots_adjust(left=0.32, right=0.89, bottom=0.18, top=0.95)
    save_figure(fig, OUT / "Fig4b_signed_attribution_distribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_c(fig, ax, grade_table, order)
    fig.subplots_adjust(left=0.32, right=0.90, bottom=0.18, top=0.95)
    save_figure(fig, OUT / "Fig4c_grade_specific_attribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_d(ax, ale, top_features)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.95)
    save_figure(fig, OUT / "Fig4d_nonlinear_ale_responses")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_e(ax, interactions, importance)
    fig.subplots_adjust(left=0.42, right=0.97, bottom=0.18, top=0.95)
    save_figure(fig, OUT / "Fig4e_interaction_excess")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    draw_panel_f(fig, ax, station_table, top_features, station_order)
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.28, top=0.95)
    save_figure(fig, OUT / "Fig4f_station_driver_heterogeneity")
    plt.close(fig)


def main() -> None:
    data, indices, audit, adjacency = load_final_data()
    feature_names = [data["features"][index] for index in indices]
    x = data["x"][:, :, indices]
    test_t = data["test_t"]
    x_test = x[test_t]
    y_test = data["y"][test_t]
    adjacency_test = adjacency[test_t]
    models, training_audit = load_or_train_models(data, x, adjacency)

    # Verify that identity-adjacency inference is node-separable before using one-node ALE/interactions.
    _, full_probability = predict_model(models[0], x_test[:1], adjacency_test[:1])
    flat_probability = predict_flat_high(models[0], x_test[0])
    if not np.allclose(full_probability[0], flat_probability, atol=1e-5):
        raise RuntimeError("No-edge model is not node-separable; one-node explanations are invalid.")

    importance_file = OUT / "panel_a_permutation_importance.csv"
    replicate_file = OUT / "panel_a_permutation_bootstrap_replicates.csv"
    if importance_file.exists() and replicate_file.exists():
        importance = pd.read_csv(importance_file, encoding="utf-8-sig")
        importance_replicates = pd.read_csv(replicate_file, encoding="utf-8-sig")
    else:
        print("Computing block-bootstrap permutation importance", flush=True)
        importance, importance_replicates = permutation_importance(
            models, x_test, y_test, adjacency_test, feature_names
        )
        importance.to_csv(importance_file, index=False, encoding="utf-8-sig")
        importance_replicates.to_csv(replicate_file, index=False, encoding="utf-8-sig")

    attribution_file = OUT / "panel_b_signed_highrisk_attributions.csv"
    if attribution_file.exists():
        attribution = pd.read_csv(attribution_file, encoding="utf-8-sig")
        mean_attribution = attribution.pivot_table(
            index=["time_id", "station_id"], columns="Feature", values="High-risk attribution"
        ).loc[:, feature_names].to_numpy().reshape(len(test_t), len(data["station_ids"]), len(feature_names))
    else:
        mean_attribution, attribution = attribution_tables(
            models, x_test, y_test, adjacency_test, feature_names, data["station_ids"],
            [data["time_ids"][index] for index in test_t],
        )
        attribution.to_csv(attribution_file, index=False, encoding="utf-8-sig")

    grade_table = (
        attribution.groupby(["Feature", "CI-CWQI grade"])["High-risk attribution"]
        .mean().unstack("CI-CWQI grade").reset_index()
    )
    grade_table.to_csv(OUT / "panel_c_grade_specific_attribution.csv", index=False, encoding="utf-8-sig")
    grade_values = grade_table.set_index("Feature")[[1, 2, 3, 4, 5]]
    grade_centered = grade_values.sub(grade_values.mean(axis=1), axis=0)
    grade_relative = grade_centered.div(grade_centered.abs().max(axis=1).replace(0, 1), axis=0)
    grade_relative.reset_index().to_csv(
        OUT / "panel_c_grade_relative_attribution_plotted.csv", index=False, encoding="utf-8-sig"
    )

    order = importance.sort_values("Mean", ascending=False)["Feature"].tolist()
    top_features = order[:6]
    ale_file = OUT / "panel_d_ale_curves.csv"
    if ale_file.exists():
        ale = pd.read_csv(ale_file, encoding="utf-8-sig")
    else:
        print("Computing ALE curves", flush=True)
        ale = ale_curves(models, x_test.reshape(-1, len(feature_names)), top_features, feature_names)
        ale.to_csv(ale_file, index=False, encoding="utf-8-sig")

    interaction_file = OUT / "panel_e_interaction_effects.csv"
    interaction_replicate_file = OUT / "panel_e_interaction_bootstrap_replicates.csv"
    if interaction_file.exists() and interaction_replicate_file.exists():
        interactions = pd.read_csv(interaction_file, encoding="utf-8-sig")
    else:
        print("Computing high-high interaction excess", flush=True)
        interactions, interaction_replicates = interaction_effects(
            models, x_test, top_features, feature_names
        )
        interactions.to_csv(interaction_file, index=False, encoding="utf-8-sig")
        interaction_replicates.to_csv(interaction_replicate_file, index=False, encoding="utf-8-sig")

    station_table = (
        attribution.groupby(["station_id", "Feature"])["High-risk attribution"]
        .mean().reset_index().rename(columns={"High-risk attribution": "Mean attribution"})
    )
    station_table.to_csv(OUT / "panel_f_station_driver_attribution.csv", index=False, encoding="utf-8-sig")
    station_probability = []
    for model in models:
        _, high = predict_model(model, x_test, adjacency_test)
        station_probability.append(high.mean(axis=0))
    station_mean_probability = np.mean(station_probability, axis=0)
    station_order = [
        station for _, station in sorted(zip(station_mean_probability, data["station_ids"]), reverse=True)
    ]
    pd.DataFrame(
        {"station_id": data["station_ids"], "Mean predicted Grade IV-V probability": station_mean_probability}
    ).to_csv(OUT / "panel_f_station_order.csv", index=False, encoding="utf-8-sig")

    audit.to_csv(OUT / "selected_indicator_audit.csv", index=False, encoding="utf-8-sig")
    draw_full_figure(
        importance, audit, attribution, grade_table, ale, interactions, station_table, station_order
    )
    draw_individual_panels(
        importance, audit, attribution, grade_table, ale, interactions, station_table, station_order
    )
    metadata = {
        "model": "GAT with identity adjacency",
        "retained_indicators": 12,
        "train_years": "2015-2021",
        "validation_year": 2022,
        "test_years": "2023-2024",
        "seeds": list(SEEDS),
        "integrated_gradient_steps": IG_STEPS,
        "permutation_bootstraps": N_PERMUTATION_BOOTSTRAPS,
        "interaction_bootstraps": N_INTERACTION_BOOTSTRAPS,
        "temporal_block_months": BLOCK_MONTHS,
    }
    (OUT / "explainability_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTop permutation importance\n", importance.head(12).to_string(index=False))
    print("\nTop interactions\n", interactions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

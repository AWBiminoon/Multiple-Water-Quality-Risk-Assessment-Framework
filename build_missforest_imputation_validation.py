"""Evaluate the retained MissForest-completed water-quality workbook.

The figure combines three complementary checks:
1) the amount of information reconstructed;
2) artificial-masking recovery relative to a median baseline; and
3) preservation of marginal distributions and pairwise correlation structure.

Artificial masking is performed before an iterative random-forest reconstruction
is fitted. The validation intentionally excludes the masked target cells from
training. It assesses imputation plausibility, not chronological leakage.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "missforest_imputation_validation_outputs"
OUT.mkdir(exist_ok=True)

RAW_FILE = Path(r"D:\星云湖\数据整合\整合水质数据_2015-2024_正确版本.xlsx")
IMPUTED_FILE = Path(r"D:\星云湖\整合水质数据_2015-2024_正确版本_missforest补齐.xlsx")

N_REPEATS = 3
MASK_FRACTION = 0.15
N_ITER = 3
N_TREES = 80
SEEDS = (2401, 2402, 2403)

BLUE = "#5E84B0"
BLUE_DARK = "#365F88"
BLUE_LIGHT = "#B5CADC"
CYAN = "#9DDFE4"
GREEN = "#AECFBC"
GREEN_DARK = "#729E88"
PURPLE = "#8D9ECF"
PINK = "#F5CED0"
GRAY = "#87939F"
PALE = "#E0E9F0"
GRID = "#D9E2EA"
TEXT = "#26313D"

META_COLUMNS = {"站点编号", "河流名称", "监测年份", "监测日期", "水体状态", "sheet"}

ENGLISH = {
    "水温": "Water temperature",
    "pH": "pH",
    "电导率": "Conductivity",
    "溶解氧": "Dissolved oxygen",
    "高锰酸盐指数": r"COD$_{Mn}$",
    "生化需氧量": r"BOD$_5$",
    "氨氮": r"NH$_4^+$-N",
    "石油类": "Petroleum",
    "挥发酚": "Volatile phenols",
    "汞": "Hg",
    "铅": "Pb",
    "化学需氧量": "COD",
    "总氮": "TN",
    "总磷": "TP",
    "铜": "Cu",
    "锌": "Zn",
    "氟化物": "Fluoride",
    "硒": "Se",
    "砷": "As",
    "镉": "Cd",
    "六价铬": "Cr(VI)",
    "氰化物": "Cyanide",
    "阴离子表面活性剂": "Surfactants",
    "硫化物": "Sulfide",
    "粪大肠菌群": "Fecal coliforms",
    "溶解氧饱和度": "DO saturation",
    "浊度": "Turbidity",
    "气压": "Air pressure",
    "气温": "Air temperature",
}

DOMAIN = {
    "水温": "Environmental factors",
    "pH": "Environmental factors",
    "溶解氧": "Environmental factors",
    "溶解氧饱和度": "Environmental factors",
    "气压": "Environmental factors",
    "气温": "Environmental factors",
    "高锰酸盐指数": "Organic pollution",
    "生化需氧量": "Organic pollution",
    "化学需氧量": "Organic pollution",
    "石油类": "Organic pollution",
    "挥发酚": "Organic pollution",
    "阴离子表面活性剂": "Organic pollution",
    "硫化物": "Organic pollution",
    "氨氮": "Nutrients",
    "总氮": "Nutrients",
    "总磷": "Nutrients",
    "汞": "Toxic metals",
    "铅": "Toxic metals",
    "铜": "Toxic metals",
    "锌": "Toxic metals",
    "氟化物": "Toxic metals",
    "硒": "Toxic metals",
    "砷": "Toxic metals",
    "镉": "Toxic metals",
    "六价铬": "Toxic metals",
    "氰化物": "Toxic metals",
    "粪大肠菌群": "Microbial",
    "电导率": "Auxiliary indicators",
    "浊度": "Auxiliary indicators",
}

DOMAIN_ORDER = [
    "Environmental factors",
    "Organic pollution",
    "Nutrients",
    "Toxic metals",
    "Microbial",
    "Auxiliary indicators",
]

DOMAIN_COLORS = {
    "Environmental factors": PURPLE,
    "Organic pollution": BLUE,
    "Nutrients": GREEN,
    "Toxic metals": CYAN,
    "Microbial": PINK,
    "Auxiliary indicators": GRAY,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def parse_numeric(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return np.nan if number == -1 else number
    text = str(value).strip().upper().replace("＜", "").replace("<", "")
    if text.endswith("L"):
        text = text[:-1]
    match = re.search(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", text)
    if not match:
        return np.nan
    number = float(match.group())
    return np.nan if number == -1 else number


def load_workbook(path: Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames: list[pd.DataFrame] = []
    for sheet_name in workbook.sheetnames:
        if sheet_name == "补齐说明":
            continue
        rows = list(workbook[sheet_name].iter_rows(min_row=3, values_only=True))
        frame = pd.DataFrame(rows[1:], columns=rows[0])
        frame.insert(0, "sheet", sheet_name)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in [item for item in data.columns if item not in META_COLUMNS]:
        data[column] = data[column].map(parse_numeric)
    data["监测年份"] = pd.to_numeric(data["监测年份"], errors="coerce")
    data["month"] = (
        data["监测日期"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    )
    return data


def select_features(raw: pd.DataFrame, completed: pd.DataFrame) -> list[str]:
    candidates = [column for column in raw.columns if column not in META_COLUMNS | {"month"}]
    return [
        column
        for column in candidates
        if column in completed
        and raw[column].notna().sum() >= 100
        and completed[column].notna().all()
        and raw[column].nunique(dropna=True) > 2
        and column in DOMAIN
    ]


def make_auxiliary_features(data: pd.DataFrame) -> np.ndarray:
    station = pd.get_dummies(data["站点编号"].astype(str), dtype=float)
    month = data["month"].fillna(1.0).to_numpy(float)
    year = data["监测年份"].fillna(data["监测年份"].median()).to_numpy(float)
    year = (year - np.nanmean(year)) / max(np.nanstd(year), 1.0)
    auxiliary = np.column_stack(
        [
            station.to_numpy(float),
            np.sin(2 * np.pi * (month - 1) / 12),
            np.cos(2 * np.pi * (month - 1) / 12),
            year,
        ]
    )
    return auxiliary


def stratified_mask(
    raw_values: np.ndarray,
    station_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    artificial = np.zeros_like(raw_values, dtype=bool)
    for variable in range(raw_values.shape[1]):
        for station in np.unique(station_ids):
            indices = np.where((station_ids == station) & np.isfinite(raw_values[:, variable]))[0]
            if len(indices) < 6:
                continue
            count = max(1, int(round(len(indices) * MASK_FRACTION)))
            count = min(count, len(indices) - 3)
            artificial[rng.choice(indices, size=count, replace=False), variable] = True
    return artificial


def initialize_matrix(masked: np.ndarray, station_ids: np.ndarray) -> np.ndarray:
    filled = masked.copy()
    global_medians = np.nanmedian(filled, axis=0)
    for variable in range(filled.shape[1]):
        for station in np.unique(station_ids):
            station_rows = station_ids == station
            station_values = filled[station_rows, variable]
            median = np.nanmedian(station_values)
            if not np.isfinite(median):
                median = global_medians[variable]
            missing = station_rows & ~np.isfinite(filled[:, variable])
            filled[missing, variable] = median
        remaining = ~np.isfinite(filled[:, variable])
        filled[remaining, variable] = global_medians[variable]
    return filled


def should_log(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) < 20 or np.nanmin(finite) < 0:
        return False
    mean = np.nanmean(finite)
    std = np.nanstd(finite)
    if std <= 0:
        return False
    skew = np.nanmean(((finite - mean) / std) ** 3)
    return bool(skew > 2.5)


def iterative_rf_impute(
    raw_values: np.ndarray,
    artificial_mask: np.ndarray,
    auxiliary: np.ndarray,
    seed: int,
) -> np.ndarray:
    masked = raw_values.copy()
    masked[artificial_mask] = np.nan
    observed_for_fit = np.isfinite(masked)
    missing_for_prediction = ~observed_for_fit
    filled = initialize_matrix(masked, station_ids_global)
    missing_fraction = missing_for_prediction.mean(axis=0)
    order = np.argsort(missing_fraction)

    for iteration in range(N_ITER):
        previous = filled.copy()
        for variable in order:
            train = observed_for_fit[:, variable]
            predict = missing_for_prediction[:, variable]
            if train.sum() < 20 or predict.sum() == 0:
                continue
            predictors = np.delete(filled, variable, axis=1)
            x = np.column_stack([predictors, auxiliary])
            y = raw_values[train, variable]
            log_target = should_log(y)
            model_y = np.log1p(np.clip(y, 0, None)) if log_target else y
            model = RandomForestRegressor(
                n_estimators=N_TREES,
                max_features="sqrt",
                min_samples_leaf=2,
                bootstrap=True,
                n_jobs=-1,
                random_state=seed + iteration * 1000 + int(variable),
            )
            model.fit(x[train], model_y)
            prediction = model.predict(x[predict])
            if log_target:
                prediction = np.expm1(prediction)
            lower, upper = np.nanmin(y), np.nanmax(y)
            filled[predict, variable] = np.clip(prediction, lower, upper)

        denominator = np.nansum(previous**2)
        change = np.nansum((filled - previous) ** 2) / max(denominator, 1e-12)
        if change < 1e-6:
            break
    return filled


def robust_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    median = np.nanmedian(reference)
    q25, q75 = np.nanpercentile(reference, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(reference)
    return (values - median) / max(scale, 1e-12)


def validation_records(
    raw: pd.DataFrame,
    features: list[str],
    auxiliary: np.ndarray,
) -> pd.DataFrame:
    raw_values = raw[features].to_numpy(float)
    station_ids = raw["站点编号"].astype(str).to_numpy()
    records: list[dict[str, object]] = []
    for repeat, seed in enumerate(SEEDS, start=1):
        rng = np.random.default_rng(seed)
        mask = stratified_mask(raw_values, station_ids, rng)
        reconstructed = iterative_rf_impute(raw_values, mask, auxiliary, seed)
        for variable_index, variable in enumerate(features):
            rows = np.where(mask[:, variable_index])[0]
            if len(rows) == 0:
                continue
            train_values = raw_values[~mask[:, variable_index] & np.isfinite(raw_values[:, variable_index]), variable_index]
            baseline = np.nanmedian(train_values)
            center = np.nanmedian(train_values)
            scale = np.nanstd(train_values)
            if not np.isfinite(scale) or scale <= 0:
                q25, q75 = np.nanpercentile(train_values, [25, 75])
                scale = max(q75 - q25, 1e-12)
            for row in rows:
                observed = raw_values[row, variable_index]
                predicted = reconstructed[row, variable_index]
                records.append(
                    {
                        "repeat": repeat,
                        "row": int(row),
                        "station": station_ids[row],
                        "year": int(raw.iloc[row]["监测年份"]),
                        "month": int(raw.iloc[row]["month"]),
                        "variable": variable,
                        "variable_label": ENGLISH[variable],
                        "domain": DOMAIN[variable],
                        "observed": observed,
                        "predicted": predicted,
                        "median_baseline": baseline,
                        "observed_scaled": (observed - center) / scale,
                        "predicted_scaled": (predicted - center) / scale,
                        "absolute_scaled_error": abs(predicted - observed) / scale,
                        "baseline_absolute_scaled_error": abs(baseline - observed) / scale,
                    }
                )
    return pd.DataFrame(records)


def summarize_validation(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (repeat, variable), group in records.groupby(["repeat", "variable"], sort=False):
        truth = group["observed"].to_numpy(float)
        prediction = group["predicted"].to_numpy(float)
        baseline = group["median_baseline"].to_numpy(float)
        scale = np.nanstd(truth)
        if not np.isfinite(scale) or scale <= 0:
            scale = max(np.nanpercentile(truth, 75) - np.nanpercentile(truth, 25), 1e-12)
        rows.append(
            {
                "repeat": repeat,
                "variable": variable,
                "variable_label": ENGLISH[variable],
                "domain": DOMAIN[variable],
                "n_masked": len(group),
                "nrmse_missforest": np.sqrt(np.mean((prediction - truth) ** 2)) / scale,
                "nrmse_median": np.sqrt(np.mean((baseline - truth) ** 2)) / scale,
                "nmae_missforest": np.mean(np.abs(prediction - truth)) / scale,
                "nmae_median": np.mean(np.abs(baseline - truth)) / scale,
                "spearman": spearmanr(truth, prediction).statistic if len(group) > 3 else np.nan,
            }
        )
    variable_metrics = pd.DataFrame(rows)
    domain_summary = (
        variable_metrics.groupby(["repeat", "domain"], as_index=False)[
            ["nrmse_missforest", "nrmse_median", "nmae_missforest", "nmae_median"]
        ]
        .mean()
    )
    station_summary = (
        records.groupby(["repeat", "station"], as_index=False)
        .agg(
            n_masked=("observed", "size"),
            nmae=("absolute_scaled_error", "mean"),
            baseline_nmae=("baseline_absolute_scaled_error", "mean"),
        )
    )
    return variable_metrics, domain_summary, station_summary


def correlation_preservation(
    raw: pd.DataFrame,
    completed: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            pair = raw[[left, right]].dropna()
            if len(pair) < 40:
                continue
            raw_r = pair.corr(method="spearman").iloc[0, 1]
            completed_r = completed[[left, right]].corr(method="spearman").iloc[0, 1]
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "n_complete_raw": len(pair),
                    "raw_spearman": raw_r,
                    "completed_spearman": completed_r,
                    "absolute_difference": abs(completed_r - raw_r),
                }
            )
    return pd.DataFrame(rows)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.07, label, transform=axis.transAxes, fontsize=9.0, fontweight="bold", va="top")


def draw_figure(
    raw: pd.DataFrame,
    completed: pd.DataFrame,
    features: list[str],
    records: pd.DataFrame,
    variable_metrics: pd.DataFrame,
    domain_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.105, top=0.975, wspace=0.38, hspace=0.42)
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]

    # a: original missingness by pressure domain, retaining variable-level points.
    ax = axes[0]
    missing = pd.DataFrame(
        {
            "variable": features,
            "missing_fraction": [raw[column].isna().mean() for column in features],
            "domain": [DOMAIN[column] for column in features],
        }
    )
    rng = np.random.default_rng(90210)
    means = []
    for position, domain in enumerate(DOMAIN_ORDER):
        values = missing.loc[missing["domain"] == domain, "missing_fraction"].to_numpy(float)
        means.append(np.mean(values))
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values * 100,
            s=13,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )
        ax.plot([position - 0.21, position + 0.21], [np.mean(values) * 100] * 2, color=TEXT, lw=1.2, zorder=4)
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels(["Environmental\nfactors", "Organic\npollution", "Nutrients", "Toxic\nmetals", "Microbial", "Auxiliary\nindicators"], rotation=35, ha="right")
    ax.set_ylabel("Original missing cells (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
    ax.text(
        0.02,
        0.97,
        f"{int(raw[features].isna().sum().sum()):,} cells reconstructed",
        transform=ax.transAxes,
        va="top",
        color=TEXT,
    )
    add_panel_label(ax, "a")

    # b: masked-cell recovery relative to a median baseline.
    ax = axes[1]
    summary = domain_summary.groupby("domain").agg(
        rf_mean=("nrmse_missforest", "mean"),
        rf_sd=("nrmse_missforest", "std"),
        median_mean=("nrmse_median", "mean"),
        median_sd=("nrmse_median", "std"),
    )
    positions = np.arange(len(DOMAIN_ORDER))[::-1]
    for position, domain in zip(positions, DOMAIN_ORDER):
        row = summary.loc[domain]
        ax.plot([row["rf_mean"], row["median_mean"]], [position, position], color=GRID, lw=1.1, zorder=1)
        ax.errorbar(row["median_mean"], position, xerr=row["median_sd"], fmt="s", ms=3.8, color=GRAY, ecolor=GRAY, capsize=1.5, lw=0.75, zorder=2)
        ax.errorbar(row["rf_mean"], position, xerr=row["rf_sd"], fmt="o", ms=4.0, color=BLUE, ecolor=BLUE, capsize=1.5, lw=0.75, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(["Environmental factors", "Organic pollution", "Nutrients", "Toxic metals", "Microbial", "Auxiliary indicators"])
    ax.set_xlabel("Masked-cell NRMSE")
    ax.grid(axis="x", color=GRID, lw=0.55, zorder=0)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="none", label="Iterative RF"),
            Line2D([], [], marker="s", color=GRAY, linestyle="none", label="Median"),
        ],
        loc="lower right",
        handletextpad=0.35,
    )
    add_panel_label(ax, "b")

    # c: pooled standardized observed-versus-reconstructed values.
    ax = axes[2]
    x = records["observed_scaled"].to_numpy(float)
    y = records["predicted_scaled"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    limit = float(max(2.5, np.nanpercentile(np.abs(np.r_[x, y]), 97.5)))
    limit = min(limit, 6.0)
    clipped_x = np.clip(x, -limit, limit)
    clipped_y = np.clip(y, -limit, limit)
    ax.hexbin(
        clipped_x,
        clipped_y,
        gridsize=28,
        mincnt=1,
        bins="log",
        cmap=mpl.colors.LinearSegmentedColormap.from_list("white_blue", ["#EDF4F8", BLUE_DARK]),
        linewidths=0,
        rasterized=True,
    )
    ax.plot([-limit, limit], [-limit, limit], color=TEXT, lw=0.8, ls="--")
    pooled_r2 = r2_score(x, y)
    pooled_rho = spearmanr(x, y).statistic
    ax.text(0.04, 0.96, f"$R^2$ = {pooled_r2:.2f}\n$\\rho$ = {pooled_rho:.2f}\n$n$ = {len(x):,}", transform=ax.transAxes, va="top", color=TEXT)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Observed value (scaled)")
    ax.set_ylabel("Reconstructed value (scaled)")
    ax.grid(color=GRID, lw=0.45, zorder=0)
    add_panel_label(ax, "c")

    # d: actual observed and imputed marginal distributions after within-variable scaling.
    ax = axes[3]
    observed_scaled: list[np.ndarray] = []
    imputed_scaled: list[np.ndarray] = []
    for variable in features:
        observed = raw[variable].to_numpy(float)
        imputed_cells = raw[variable].isna().to_numpy() & completed[variable].notna().to_numpy()
        reference = observed[np.isfinite(observed)]
        observed_scaled.append(robust_scale(reference, reference))
        imputed_scaled.append(robust_scale(completed.loc[imputed_cells, variable].to_numpy(float), reference))
    observed_values = np.concatenate(observed_scaled)
    imputed_values = np.concatenate(imputed_scaled)
    display_limit = 4.0
    bins = np.linspace(-display_limit, display_limit, 61)
    obs_density, edges = np.histogram(np.clip(observed_values, -display_limit, display_limit), bins=bins, density=True)
    imp_density, _ = np.histogram(np.clip(imputed_values, -display_limit, display_limit), bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, 0, obs_density, color=BLUE, alpha=0.18)
    ax.plot(centers, obs_density, color=BLUE_DARK, label="Observed")
    ax.fill_between(centers, 0, imp_density, color=GREEN, alpha=0.25)
    ax.plot(centers, imp_density, color=GREEN_DARK, label="Imputed")
    ax.set_xlabel("Within-indicator robust z-score")
    ax.set_ylabel("Density")
    ax.set_xlim(-display_limit, display_limit)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.legend(loc="upper right", ncol=1)
    add_panel_label(ax, "d")

    # e: preservation of pairwise rank correlations.
    ax = axes[4]
    corr_x = correlations["raw_spearman"].to_numpy(float)
    corr_y = correlations["completed_spearman"].to_numpy(float)
    ax.scatter(corr_x, corr_y, s=12, color=BLUE, alpha=0.58, edgecolor="white", linewidth=0.25)
    ax.plot([-1, 1], [-1, 1], color=TEXT, lw=0.8, ls="--")
    concordance = np.corrcoef(corr_x, corr_y)[0, 1]
    corr_mae = np.mean(np.abs(corr_y - corr_x))
    ax.text(0.04, 0.96, f"$r$ = {concordance:.2f}\nMAE = {corr_mae:.2f}\n$n$ = {len(correlations)} pairs", transform=ax.transAxes, va="top")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Observed-pair Spearman $\\rho$")
    ax.set_ylabel("Completed-data Spearman $\\rho$")
    ax.grid(color=GRID, lw=0.45)
    add_panel_label(ax, "e")

    # f: station-level masking errors and between-repeat variation.
    ax = axes[5]
    station_stats = station_summary.groupby("station").agg(mean=("nmae", "mean"), sd=("nmae", "std")).sort_values("mean")
    y_positions = np.arange(len(station_stats))
    ax.errorbar(station_stats["mean"], y_positions, xerr=station_stats["sd"], fmt="o", color=BLUE, ecolor=BLUE_LIGHT, ms=3.5, capsize=1.4, lw=0.75)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(station_stats.index)
    ax.set_xlabel("Station-level normalized MAE")
    ax.grid(axis="x", color=GRID, lw=0.55)
    add_panel_label(ax, "f")

    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    global station_ids_global
    raw = load_workbook(RAW_FILE)
    completed = load_workbook(IMPUTED_FILE)
    # Cells altered between the pre-imputation source and retained completed
    # workbook are not valid masking ground truth. They are retained in a
    # separate audit table and treated as missing for validation.
    audit_rows: list[dict[str, object]] = []
    preliminary_features = select_features(raw, completed)
    for variable in preliminary_features:
        changed = (
            raw[variable].notna()
            & completed[variable].notna()
            & ~np.isclose(raw[variable], completed[variable], rtol=0, atol=0, equal_nan=True)
        )
        for index in np.where(changed.to_numpy())[0]:
            audit_rows.append(
                {
                    "row": int(index),
                    "station": raw.iloc[index]["站点编号"],
                    "year": raw.iloc[index]["监测年份"],
                    "month": raw.iloc[index]["month"],
                    "variable": variable,
                    "source_value": raw.iloc[index][variable],
                    "completed_workbook_value": completed.iloc[index][variable],
                }
            )
        raw.loc[changed, variable] = np.nan
    pd.DataFrame(audit_rows).to_csv(
        OUT / "qc_flagged_source_cells.csv", index=False, encoding="utf-8-sig"
    )
    features = select_features(raw, completed)
    station_ids_global = raw["站点编号"].astype(str).to_numpy()
    auxiliary = make_auxiliary_features(raw)

    missingness = pd.DataFrame(
        {
            "variable": features,
            "variable_label": [ENGLISH[column] for column in features],
            "domain": [DOMAIN[column] for column in features],
            "n_observed": [raw[column].notna().sum() for column in features],
            "n_imputed": [raw[column].isna().sum() for column in features],
            "missing_fraction": [raw[column].isna().mean() for column in features],
        }
    )
    missingness.to_csv(OUT / "selected_variable_missingness.csv", index=False, encoding="utf-8-sig")

    records = validation_records(raw, features, auxiliary)
    records.to_csv(OUT / "masked_cell_predictions.csv", index=False, encoding="utf-8-sig")
    variable_metrics, domain_summary, station_summary = summarize_validation(records)
    variable_metrics.to_csv(OUT / "variable_validation_metrics.csv", index=False, encoding="utf-8-sig")
    domain_summary.to_csv(OUT / "domain_validation_summary.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(OUT / "station_validation_summary.csv", index=False, encoding="utf-8-sig")

    correlations = correlation_preservation(raw, completed, features)
    correlations.to_csv(OUT / "correlation_preservation.csv", index=False, encoding="utf-8-sig")

    observed_exact = []
    for variable in features:
        observed = raw[variable].notna()
        observed_exact.append(np.allclose(raw.loc[observed, variable], completed.loc[observed, variable], rtol=0, atol=0))
    overall = pd.DataFrame(
        [
            {
                "n_station_months": len(raw),
                "n_variables_evaluated": len(features),
                "n_observed_cells": int(raw[features].notna().sum().sum()),
                "n_imputed_cells": int(raw[features].isna().sum().sum()),
                "mean_missing_fraction": float(raw[features].isna().mean().mean()),
                "n_qc_flagged_source_cells": len(audit_rows),
                "observed_values_preserved_exactly": bool(all(observed_exact)),
                "masked_cells_evaluated": len(records),
                "pooled_scaled_r2": float(r2_score(records["observed_scaled"], records["predicted_scaled"])),
                "pooled_scaled_spearman": float(spearmanr(records["observed_scaled"], records["predicted_scaled"]).statistic),
                "mean_variable_nrmse_missforest": float(variable_metrics["nrmse_missforest"].mean()),
                "mean_variable_nrmse_median": float(variable_metrics["nrmse_median"].mean()),
                "correlation_preservation_r": float(np.corrcoef(correlations["raw_spearman"], correlations["completed_spearman"])[0, 1]),
                "correlation_absolute_error": float(correlations["absolute_difference"].mean()),
            }
        ]
    )
    overall.to_csv(OUT / "overall_validation_summary.csv", index=False, encoding="utf-8-sig")

    fig = draw_figure(raw, completed, features, records, variable_metrics, domain_summary, station_summary, correlations)
    save_figure(fig, OUT / "Figure_S1_MissForest_imputation_validation")
    plt.close(fig)
    print(overall.to_string(index=False))
    print("\nDomain NRMSE means")
    print(domain_summary.groupby("domain")[["nrmse_missforest", "nrmse_median"]].mean().to_string())


if __name__ == "__main__":
    main()

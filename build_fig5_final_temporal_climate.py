from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CI_FILE = (
    ROOT
    / "cluster_informed_cwqi_kmeans_outputs"
    / "cluster_informed_cwqi_scores_and_grades.csv"
)
WEIGHT_FILE = (
    ROOT
    / "cluster_informed_cwqi_kmeans_outputs"
    / "cluster_informed_cwqi_weights.csv"
)
PANEL_FILE = ROOT / "viz_input" / "gnn_water_quality_panel.csv"
RAINFALL_FILE = Path(r"D:\星云湖\降水数据 月.xlsx")
OUT = ROOT / "fig5_final_temporal_climate_outputs"
ANNUAL_INTERACTION_FILE = OUT / "Fig5f_annual_key_interaction_source_data.csv"
INTERACTION_CONTRAST_FILE = OUT / "Fig5f_key_interaction_period_contrasts.csv"

TEXT = "#26313D"
MUTED = "#607080"
GRID = "#D9E2EA"
WHITE = "#FFFFFF"
ZERO = "#7E8994"

COMPONENTS = [
    ("oxygen", "Oxygen–acid-base", "Oxygen-acid-base", "#8D9ECF"),
    ("organic", "Organic oxygen demand", "Organic-oxygen", "#4D77A7"),
    ("nutrients", "Nutrients", "Nutrients", "#A5CAB5"),
    ("toxic", "Toxic–metal", "Toxic-metal", "#9DDFE4"),
    ("microbial", "Microbial", "Microbial", "#F5CED0"),
]
CLIMATE = [
    ("precipitation", "Precipitation", "#4D77A7", "o"),
    ("air_temperature", "Air temperature", "#C98270", "s"),
    ("water_temperature", "Water temperature", "#6FAFA3", "D"),
]
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_MAP = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
    12: "Winter",
}
BOOT_REPEATS = 1600
BLOCK_LENGTH = 12
BASE_SEED = 20260731


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def seasonal_anomaly(values: pd.Series, month: pd.Series) -> pd.Series:
    centered = values - values.groupby(month).transform("mean")
    return zscore(centered)


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def read_rainfall() -> pd.DataFrame:
    table = pd.read_excel(RAINFALL_FILE, sheet_name="Sheet1")
    if len(table) != 1:
        raise ValueError("Expected one county row in the rainfall workbook.")
    values = table.iloc[0]
    records = []
    for column in table.columns:
        label = str(column)
        if len(label) == 7 and label[4] == "-" and label[:4].isdigit():
            year = int(label[:4])
            month = int(label[5:])
            if 2015 <= year <= 2024:
                records.append(
                    {
                        "time_id": label,
                        "year": year,
                        "month": month,
                        "precipitation": float(values[column]),
                    }
                )
    rainfall = pd.DataFrame(records).sort_values(["year", "month"])
    if len(rainfall) != 120 or rainfall["precipitation"].isna().any():
        raise ValueError("Rainfall series must contain 120 complete months.")
    return rainfall


def prepare_monthly_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(CI_FILE, encoding="utf-8-sig")
    weights = pd.read_csv(WEIGHT_FILE, encoding="utf-8-sig")
    panel = pd.read_csv(PANEL_FILE, encoding="utf-8-sig")
    rainfall = read_rainfall()

    weight_lookup = weights.set_index("Component")["Cluster-informed CWQI"]
    contribution_columns = []
    for code, source_column, _, _ in COMPONENTS:
        column = f"{code}_contribution"
        labels[column] = labels[source_column] * float(weight_lookup[source_column])
        contribution_columns.append(column)

    monthly = (
        labels.groupby(["time_id", "year", "month"], as_index=False)
        .agg(
            **{column: (column, "mean") for column in contribution_columns},
            mean_ci_cwqi=("Cluster-informed CWQI score", "mean"),
            high_risk_fraction=(
                "Cluster-informed CWQI grade",
                lambda values: float(np.mean(values >= 4)),
            ),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    climate_monthly = (
        panel.groupby(["time_id", "year", "month"], as_index=False)
        .agg(
            air_temperature=("气温", "mean"),
            water_temperature=("水温", "mean"),
            air_pressure=("气压", "mean"),
        )
    )
    monthly = monthly.merge(climate_monthly, on=["time_id", "year", "month"], how="left")
    monthly = monthly.merge(rainfall, on=["time_id", "year", "month"], how="left")
    monthly["date"] = pd.to_datetime(monthly["time_id"] + "-01")
    monthly["season"] = monthly["month"].map(SEASON_MAP)
    monthly["trend"] = np.linspace(-1.0, 1.0, len(monthly))

    required = [
        "precipitation",
        "air_temperature",
        "water_temperature",
        "mean_ci_cwqi",
        "high_risk_fraction",
    ] + contribution_columns
    if monthly[required].isna().any().any():
        missing = monthly[required].isna().sum()
        raise ValueError(f"Missing monthly values:\n{missing[missing > 0]}")

    for column in required:
        monthly[f"{column}_z"] = zscore(monthly[column])
        monthly[f"{column}_anomaly_z"] = seasonal_anomaly(
            monthly[column], monthly["month"]
        )

    annual = (
        monthly.groupby("year", as_index=False)
        .agg(
            **{column: (column, "mean") for column in contribution_columns},
            mean_ci_cwqi=("mean_ci_cwqi", "mean"),
            high_risk_fraction=("high_risk_fraction", "mean"),
            annual_precipitation=("precipitation", "sum"),
            mean_air_temperature=("air_temperature", "mean"),
        )
    )
    total = annual[contribution_columns].sum(axis=1)
    for code, _, _, _ in COMPONENTS:
        annual[f"{code}_share"] = annual[f"{code}_contribution"] / total
    shares = [f"{code}_share" for code, _, _, _ in COMPONENTS]
    ranks = annual[shares].rank(axis=1, method="min", ascending=False)
    ranks.columns = [f"{code}_rank" for code, _, _, _ in COMPONENTS]
    annual = pd.concat([annual, ranks], axis=1)

    season_year = (
        monthly.groupby(["year", "season"], as_index=False)
        .agg(
            **{column: (column, "mean") for column in contribution_columns},
            mean_ci_cwqi=("mean_ci_cwqi", "mean"),
        )
    )
    return monthly, annual, season_year


def control_matrix(month: pd.Series, trend: pd.Series) -> np.ndarray:
    month_dummy = pd.get_dummies(month.astype(int), drop_first=True, dtype=float)
    return np.column_stack(
        [
            np.ones(len(month)),
            np.asarray(trend, dtype=float),
            month_dummy.to_numpy(dtype=float),
        ]
    )


def adjusted_beta(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([controls, x])
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficient[-1])


def moving_block_indices(
    n: int,
    block_length: int,
    repeats: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    samples = []
    for _ in range(repeats):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_length)) % n for start in starts]
        )[:n]
        samples.append(indices)
    return samples


def bootstrap_adjusted_beta(
    x: pd.Series,
    y: pd.Series,
    month: pd.Series,
    trend: pd.Series,
    seed: int,
) -> tuple[float, float, float, float]:
    table = pd.DataFrame(
        {"x": x, "y": y, "month": month, "trend": trend}
    ).dropna()
    x_values = zscore(table["x"]).to_numpy()
    y_values = zscore(table["y"]).to_numpy()
    controls = control_matrix(table["month"], table["trend"])
    estimate = adjusted_beta(x_values, y_values, controls)
    bootstrap = []
    for indices in moving_block_indices(
        len(table), BLOCK_LENGTH, BOOT_REPEATS, seed
    ):
        value = adjusted_beta(
            x_values[indices], y_values[indices], controls[indices]
        )
        if np.isfinite(value):
            bootstrap.append(value)
    bootstrap_values = np.asarray(bootstrap)
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    p_value = 2 * min(
        np.mean(bootstrap_values <= 0), np.mean(bootstrap_values >= 0)
    )
    p_value = max(float(p_value), 1.0 / len(bootstrap_values))
    return estimate, float(lower), float(upper), p_value


def compute_adjusted_associations(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed = BASE_SEED
    for code, _, label, _ in COMPONENTS:
        for climate_code, climate_label, _, _ in CLIMATE:
            estimate, lower, upper, p_value = bootstrap_adjusted_beta(
                monthly[climate_code],
                monthly[f"{code}_contribution"],
                monthly["month"],
                monthly["trend"],
                seed,
            )
            seed += 1
            rows.append(
                {
                    "component_code": code,
                    "component": label,
                    "climate_code": climate_code,
                    "climate_variable": climate_label,
                    "standardized_beta": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_p": p_value,
                    "n_lake_months": len(monthly),
                    "controls": "calendar month fixed effects + linear trend",
                    "bootstrap": f"circular moving blocks, length={BLOCK_LENGTH}",
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = fdr_bh(result["bootstrap_p"].to_numpy())
    return result


def compute_lagged_associations(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed = BASE_SEED + 1000
    for code, _, label, _ in COMPONENTS:
        response = monthly[f"{code}_contribution"]
        for lag in range(7):
            predictor = monthly["precipitation"].shift(lag)
            estimate, lower, upper, p_value = bootstrap_adjusted_beta(
                predictor,
                response,
                monthly["month"],
                monthly["trend"],
                seed,
            )
            seed += 1
            rows.append(
                {
                    "component_code": code,
                    "component": label,
                    "precipitation_lead_months": lag,
                    "standardized_beta": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_p": p_value,
                    "n_lake_months": int(len(monthly) - lag),
                    "controls": "calendar month fixed effects + linear trend",
                    "bootstrap": f"circular moving blocks, length={BLOCK_LENGTH}",
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = fdr_bh(result["bootstrap_p"].to_numpy())
    return result


def bootstrap_season_totals(season_year: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BASE_SEED + 2000)
    rows = []
    for season in SEASON_ORDER:
        values = season_year.loc[
            season_year["season"] == season, "mean_ci_cwqi"
        ].to_numpy()
        draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "season": season,
                "mean_ci_cwqi": float(values.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_years": len(values),
            }
        )
    return pd.DataFrame(rows)


def compute_period_source_shifts(monthly: pd.DataFrame) -> pd.DataFrame:
    table = monthly.copy()
    contribution_columns = [f"{code}_contribution" for code, _, _, _ in COMPONENTS]
    total = table[contribution_columns].sum(axis=1)
    rows = []
    for index, (code, _, label, _) in enumerate(COMPONENTS):
        share = 100 * table[f"{code}_contribution"] / total
        early = share[table["year"] <= 2019].to_numpy(float)
        late = share[table["year"] >= 2020].to_numpy(float)
        estimate = float(late.mean() - early.mean())
        early_draws = moving_block_indices(
            len(early), BLOCK_LENGTH, BOOT_REPEATS, BASE_SEED + 3100 + 2 * index
        )
        late_draws = moving_block_indices(
            len(late), BLOCK_LENGTH, BOOT_REPEATS, BASE_SEED + 3101 + 2 * index
        )
        bootstrap = np.asarray(
            [late[j].mean() - early[i].mean() for i, j in zip(early_draws, late_draws)]
        )
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        p_value = 2 * min(np.mean(bootstrap <= 0), np.mean(bootstrap >= 0))
        rows.append(
            {
                "component_code": code,
                "component": label,
                "2015-2019 mean share (%)": float(early.mean()),
                "2020-2024 mean share (%)": float(late.mean()),
                "change (percentage points)": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "p_value": max(float(p_value), 1.0 / BOOT_REPEATS),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = fdr_bh(result["p_value"].to_numpy())
    return result


def draw_panel_a(ax: plt.Axes, period_shifts: pd.DataFrame) -> None:
    lookup = period_shifts.set_index("component_code")
    order = [code for code, _, _, _ in COMPONENTS]
    labels = [label for _, _, label, _ in COMPONENTS]
    colors = [color for _, _, _, color in COMPONENTS]
    y = np.arange(len(order))
    estimate = lookup.loc[order, "change (percentage points)"].to_numpy(float)
    lower = estimate - lookup.loc[order, "ci_lower"].to_numpy(float)
    upper = lookup.loc[order, "ci_upper"].to_numpy(float) - estimate
    significant = lookup.loc[order, "q_value"].to_numpy(float) < 0.05
    ax.axvline(0, color=ZERO, linewidth=0.8, zorder=1)
    for idx, (value, lo, hi, color, is_significant) in enumerate(
        zip(estimate, lower, upper, colors, significant)
    ):
        ax.errorbar(
            value,
            idx,
            xerr=np.asarray([[lo], [hi]]),
            fmt="o",
            color=color,
            markerfacecolor=color if is_significant else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=4.4,
            elinewidth=0.8,
            capsize=1.8,
            zorder=3,
        )
        ax.annotate(
            f"{value:+.1f}",
            (value, idx),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.7,
            color=TEXT,
        )
    span = max(abs(lookup["ci_lower"]).max(), abs(lookup["ci_upper"]).max())
    ax.set_xlim(-1.12 * span, 1.12 * span)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Change in relative contribution (pp)\n2020-2024 minus 2015-2019"
    )
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)


def draw_panel_b(ax: plt.Axes, annual: pd.DataFrame) -> None:
    years = annual["year"].to_numpy()
    values = [annual[f"{code}_share"].to_numpy() for code, _, _, _ in COMPONENTS]
    ax.stackplot(
        years,
        values,
        colors=[color for _, _, _, color in COMPONENTS],
        edgecolor=WHITE,
        linewidth=0.5,
    )
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 1)
    ax.set_xticks(years[::2])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative contribution (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.55)


def draw_panel_c(
    ax: plt.Axes,
    season_year: pd.DataFrame,
    season_ci: pd.DataFrame,
) -> None:
    component_columns = [f"{code}_contribution" for code, _, _, _ in COMPONENTS]
    means = (
        season_year.groupby("season")[component_columns]
        .mean()
        .reindex(SEASON_ORDER)
    )
    x = np.arange(len(SEASON_ORDER))
    bottom = np.zeros(len(SEASON_ORDER))
    for code, _, label, color in COMPONENTS:
        values = means[f"{code}_contribution"].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            label=label,
            zorder=2,
        )
        bottom += values
    lower = season_ci["mean_ci_cwqi"] - season_ci["ci_lower"]
    upper = season_ci["ci_upper"] - season_ci["mean_ci_cwqi"]
    ax.errorbar(
        x,
        season_ci["mean_ci_cwqi"],
        yerr=np.vstack([lower, upper]),
        fmt="o",
        color=TEXT,
        markerfacecolor=WHITE,
        markeredgewidth=0.8,
        markersize=3.6,
        linewidth=0.8,
        capsize=2,
        zorder=4,
    )
    ax.set_xticks(x, SEASON_ORDER)
    ax.tick_params(axis="x", labelsize=5.6, pad=2)
    ax.set_ylabel("Mean weighted contribution")
    ax.set_xlabel("Season")
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)


def draw_panel_d(ax: plt.Axes, monthly: pd.DataFrame) -> None:
    smooth_window = 6
    series = [
        (
            monthly["high_risk_fraction_anomaly_z"].rolling(
                smooth_window, center=True, min_periods=3
            ).mean(),
            "Grade IV-V burden",
            "#26313D",
            1.45,
        ),
        (
            monthly["precipitation_anomaly_z"].rolling(
                smooth_window, center=True, min_periods=3
            ).mean(),
            "Precipitation",
            "#4D77A7",
            1.15,
        ),
        (
            monthly["air_temperature_anomaly_z"].rolling(
                smooth_window, center=True, min_periods=3
            ).mean(),
            "Air temperature",
            "#C98270",
            1.15,
        ),
    ]
    ax.axhline(0, color=ZERO, linewidth=0.7, zorder=1)
    for values, label, color, width in series:
        ax.plot(monthly["date"], values, color=color, linewidth=width, label=label, zorder=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(monthly["date"].min(), monthly["date"].max())
    ax.set_xlabel("Lake month")
    ax.set_ylabel("Deseasonalized anomaly (Z)")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(loc="lower left", fontsize=6.2, ncol=1, handlelength=1.8)
    ax.text(
        0.98,
        0.98,
        "6-month moving average",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=MUTED,
    )


def draw_panel_e(ax: plt.Axes, associations: pd.DataFrame) -> None:
    order = [code for code, _, _, _ in COMPONENTS]
    labels = [label for _, _, label, _ in COMPONENTS]
    y_base = np.arange(len(order))
    offsets = [-0.20, 0.0, 0.20]
    ax.axvline(0, color=ZERO, linewidth=0.8, zorder=1)
    for offset, (climate_code, climate_label, color, marker) in zip(offsets, CLIMATE):
        subset = (
            associations[associations["climate_code"] == climate_code]
            .set_index("component_code")
            .reindex(order)
        )
        estimate = subset["standardized_beta"].to_numpy()
        lower = estimate - subset["ci_lower"].to_numpy()
        upper = subset["ci_upper"].to_numpy() - estimate
        significant = subset["q_value"].to_numpy() < 0.05
        ax.errorbar(
            estimate,
            y_base + offset,
            xerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=1.8,
            zorder=2,
        )
        ax.scatter(
            estimate,
            y_base + offset,
            s=24,
            marker=marker,
            facecolor=np.where(significant, color, WHITE),
            edgecolor=color,
            linewidth=0.8,
            label=climate_label,
            zorder=3,
        )
    ax.set_yticks(y_base, labels)
    ax.set_ylim(len(order) - 0.55, -0.55)
    ax.set_xlabel("Adjusted beta per 1 SD climate anomaly")
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.legend(loc="upper right", fontsize=5.0, handletextpad=0.35, labelspacing=0.25)
    ax.text(
        0.98,
        0.02,
        "filled: BH-FDR q < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.7,
        color=MUTED,
    )


def draw_panel_f(
    ax: plt.Axes,
    annual_interactions: pd.DataFrame,
    interaction_contrasts: pd.DataFrame,
) -> None:
    colors = ["#4D77A7", "#8D9ECF", "#A5CAB5"]
    offsets = [-0.10, 0.0, 0.10]
    pair_display = {
        "NH$_4^+$-N x TN": r"NH$_4^+$-N $\times$ TN",
        "BOD$_5$ x TN": r"BOD$_5$ $\times$ TN",
        "TP x TN": r"TP $\times$ TN",
    }
    pair_order = annual_interactions["Pair"].drop_duplicates().tolist()
    contrast_lookup = interaction_contrasts.set_index("Pair")
    ax.axhline(0, color=ZERO, linewidth=0.7, zorder=1)
    for pair, color, offset in zip(pair_order, colors, offsets):
        subset = annual_interactions[annual_interactions["Pair"] == pair].sort_values(
            "Year"
        )
        x = subset["Year"].to_numpy(float) + offset
        estimate = subset[
            "Mean interaction excess (percentage points)"
        ].to_numpy(float)
        lower = estimate - subset["CI low"].to_numpy(float)
        upper = subset["CI high"].to_numpy(float) - estimate
        change = float(
            contrast_lookup.loc[
                pair, "Late-minus-early change (percentage points)"
            ]
        )
        display_pair = pair_display.get(pair, pair.replace(" x ", " × "))
        ax.errorbar(
            x,
            estimate,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker="o",
            markersize=2.8,
            markeredgecolor=WHITE,
            markeredgewidth=0.35,
            linewidth=1.1,
            elinewidth=0.45,
            capsize=1.0,
            alpha=0.96,
            label=f"{display_pair}  Δ={change:+.1f}",
            zorder=2,
        )
    years = np.sort(annual_interactions["Year"].unique())
    ax.set_xlim(years.min() - 0.35, years.max() + 0.35)
    ax.set_xticks(years[::2])
    ax.set_xlabel("Year")
    ax.set_ylabel("Interaction excess (percentage points)")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(
        loc="upper left",
        fontsize=4.25,
        handlelength=1.45,
        labelspacing=0.25,
        borderaxespad=0.15,
    )
    ax.text(
        0.98,
        0.02,
        "Δ: 2020-2024 minus 2015-2019; all q ≥ 0.21",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.3,
        color=MUTED,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def save_fixed_4_3(fig: plt.Figure, stem: Path) -> None:
    """Export the nominal 7.2 x 5.4 inch canvas without tight cropping."""
    fig.savefig(stem.with_suffix(".svg"), facecolor=WHITE)
    fig.savefig(stem.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor=WHITE)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_figure(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    period_shifts: pd.DataFrame,
    season_year: pd.DataFrame,
    season_ci: pd.DataFrame,
    associations: pd.DataFrame,
    lagged: pd.DataFrame,
    annual_interactions: pd.DataFrame,
    interaction_contrasts: pd.DataFrame,
) -> None:
    set_style()
    fig = plt.figure(figsize=(7.2, 5.4), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.90,
        wspace=0.54,
        hspace=0.48,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    draw_panel_b(axes[0], annual)
    draw_panel_a(axes[1], period_shifts)
    draw_panel_c(axes[2], season_year, season_ci)
    draw_panel_d(axes[3], monthly)
    draw_panel_e(axes[4], associations)
    draw_panel_f(axes[5], annual_interactions, interaction_contrasts)

    source_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.25,
            label=label,
        )
        for _, _, label, color in COMPONENTS
    ]
    fig.legend(
        handles=source_handles,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        columnspacing=0.85,
        handlelength=1.35,
        fontsize=5.4,
    )
    for panel, axis in zip("abcdef", axes):
        add_panel_label(axis, panel)
    save_figure(fig, OUT / "Fig5_temporal_climate_risk_contributions_3x2")
    save_fixed_4_3(fig, OUT / "Fig5_FINAL_temporal_climate_contributions_3x2")
    plt.close(fig)


def save_source_data(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    period_shifts: pd.DataFrame,
    season_year: pd.DataFrame,
    season_ci: pd.DataFrame,
    associations: pd.DataFrame,
    lagged: pd.DataFrame,
) -> None:
    monthly.to_csv(OUT / "Fig5d_monthly_lake_level_source_data.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "Fig5ab_annual_source_data.csv", index=False, encoding="utf-8-sig")
    period_shifts.to_csv(
        OUT / "Fig5b_period_source_shifts.csv", index=False, encoding="utf-8-sig"
    )
    season_year.to_csv(OUT / "Fig5c_year_by_season_source_data.csv", index=False, encoding="utf-8-sig")
    season_ci.to_csv(OUT / "Fig5c_seasonal_total_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(OUT / "Fig5e_adjusted_climate_source_associations.csv", index=False, encoding="utf-8-sig")
    lagged.to_csv(OUT / "Fig5f_adjusted_lagged_precipitation_associations.csv", index=False, encoding="utf-8-sig")


def write_methods_note(
    associations: pd.DataFrame,
    lagged: pd.DataFrame,
    interaction_contrasts: pd.DataFrame,
    period_shifts: pd.DataFrame,
) -> None:
    significant = associations[associations["q_value"] < 0.05]
    lag_significant = lagged[lagged["q_value"] < 0.05]
    interaction_significant = interaction_contrasts[
        interaction_contrasts["q value"] < 0.05
    ]
    text = f"""# Figure 5 source-data and statistical note

## Figure contract

Core conclusion: CI-CWQI risk-source composition varies across years and seasons, while climate covariates are associated with selected components at the lake-month scale.

Archetype: 2 x 3 quantitative evidence grid.

## Data unit and construction

- The climate-analysis unit is one lake month (n = 120; January 2015 to December 2024).
- Monthly precipitation is a lake-wide exposure and is not replicated across the 16 stations for inference.
- CI-CWQI component contributions are the station-month component scores multiplied by the final CI-CWQI component weights and then averaged across stations within each month.
- Grade IV-V burden is the monthly fraction of stations assigned to CI-CWQI Grades IV or V.

## Statistical intervals

- Panel c: seasonal total CI-CWQI means and 95% nonparametric bootstrap intervals across 10 yearly seasonal means.
- Panel b: the 2020-2024 minus 2015-2019 change in the monthly relative contribution of each CI-CWQI component. Intervals are percentile 95% CIs from {BOOT_REPEATS} independent circular moving-block bootstrap samples within each five-year period (block length = {BLOCK_LENGTH} months); Benjamini-Hochberg correction is applied across the five components.
- Panel e: standardized OLS coefficients adjusted for calendar-month fixed effects and a linear time trend. Intervals are percentile 95% CIs from {BOOT_REPEATS} circular moving-block bootstrap samples (block length = {BLOCK_LENGTH} months). Benjamini-Hochberg q values are calculated across the 15 climate-component tests.
- Panel f: the final five-seed GraphSAGE ensemble was held fixed, and the three strongest full-data interactions from Figure 4 were evaluated within each calendar year using common full-period 50th and 80th percentile intervention values. Annual intervals are percentile 95% CIs from {BOOT_REPEATS} circular moving-block bootstrap samples (block length = 3 months). The displayed period contrast is the 2020-2024 mean minus the 2015-2019 mean; its interval uses 12-month circular moving blocks, with Benjamini-Hochberg correction across the three prespecified pairs.
- The precipitation lag analysis is retained as a supporting analysis rather than a main-text panel. Each lag is estimated in a separate adjusted model using precipitation at t-lag and component contribution at t. Benjamini-Hochberg q values are calculated across the 35 lag-component tests.
- Filled markers in panel e denote q < 0.05; open markers denote q >= 0.05.

## Interpretation limits

- Panels e and f show adjusted temporal associations, not causal effects.
- Water temperature is a proximal lake-state covariate and should not be interpreted as an independent external forcing in the same way as precipitation or air temperature.
- Significant panel-e associations: {len(significant)} of 15.
- Significant panel-f lag associations: {len(lag_significant)} of 35.
- Significant early-versus-late interaction changes: {len(interaction_significant)} of 3.
- Significant early-versus-late component-share changes: {int((period_shifts['q_value'] < 0.05).sum())} of 5.
"""
    (OUT / "Fig5_statistical_methods_and_QA.md").write_text(text, encoding="utf-8")


def print_summary(
    annual: pd.DataFrame,
    associations: pd.DataFrame,
    lagged: pd.DataFrame,
) -> None:
    print("Annual source ranks")
    columns = ["year"] + [f"{code}_rank" for code, _, _, _ in COMPONENTS]
    print(annual[columns].to_string(index=False))
    print("\nAdjusted concurrent associations sorted by absolute beta")
    print(
        associations.assign(abs_beta=lambda x: x["standardized_beta"].abs())
        .sort_values("abs_beta", ascending=False)[
            [
                "component",
                "climate_variable",
                "standardized_beta",
                "ci_lower",
                "ci_upper",
                "q_value",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )
    print("\nStrongest precipitation lag by component")
    strongest = (
        lagged.assign(abs_beta=lambda x: x["standardized_beta"].abs())
        .sort_values("abs_beta", ascending=False)
        .groupby("component", as_index=False)
        .first()
    )
    print(
        strongest[
            [
                "component",
                "precipitation_lead_months",
                "standardized_beta",
                "ci_lower",
                "ci_upper",
                "q_value",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    monthly, annual, season_year = prepare_monthly_data()
    period_shifts = compute_period_source_shifts(monthly)
    season_ci = bootstrap_season_totals(season_year)
    associations = compute_adjusted_associations(monthly)
    lagged = compute_lagged_associations(monthly)
    if not ANNUAL_INTERACTION_FILE.exists() or not INTERACTION_CONTRAST_FILE.exists():
        raise FileNotFoundError(
            "Run build_fig5_annual_interaction_dynamics.py before drawing Figure 5."
        )
    annual_interactions = pd.read_csv(ANNUAL_INTERACTION_FILE, encoding="utf-8-sig")
    interaction_contrasts = pd.read_csv(
        INTERACTION_CONTRAST_FILE, encoding="utf-8-sig"
    )
    save_source_data(
        monthly, annual, period_shifts, season_year, season_ci, associations, lagged
    )
    write_methods_note(associations, lagged, interaction_contrasts, period_shifts)
    draw_figure(
        monthly,
        annual,
        period_shifts,
        season_year,
        season_ci,
        associations,
        lagged,
        annual_interactions,
        interaction_contrasts,
    )
    print_summary(annual, associations, lagged)


if __name__ == "__main__":
    main()

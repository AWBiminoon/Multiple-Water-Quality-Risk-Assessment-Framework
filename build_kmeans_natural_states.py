"""Build the management-compatible five-state KMeans reference.

Cluster membership is determined only from the stored PCA coordinates. Raw
cluster IDs are ordered afterward by the cluster-level median legacy CWQI so
that state numbering has a transparent low-to-high management interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


ROOT = Path(__file__).resolve().parent
ASSIGNMENT_INPUT = ROOT / "viz_input" / "best_dimred_cluster_assignments.csv"
SCORE_INPUT = ROOT / "enhanced_cwqi_outputs" / "enhanced_cwqi_scores.csv"
ASSIGNMENT_OUTPUT = ROOT / "viz_input" / "best_dimred_cluster_assignments_kmeans.csv"
SCORE_OUTPUT = ROOT / "enhanced_cwqi_outputs" / "enhanced_cwqi_scores_kmeans.csv"
VALIDATION_OUTPUT = ROOT / "cluster_validation_kmeans_outputs"
VALIDATION_OUTPUT.mkdir(exist_ok=True)

N_STATES = 5
N_BOOTSTRAP = 200
SEED = 42

STATE_NAMES = {
    1: "Natural state I",
    2: "Natural state II",
    3: "Natural state III",
    4: "Natural state IV",
    5: "Natural state V",
}

GROUPS = {
    "Oxygen–acid-base": ["pH_分指数", "溶解氧_分指数"],
    "Organic–oxygen": [
        "高锰酸盐指数_分指数",
        "化学需氧量_分指数",
        "生化需氧量_分指数",
    ],
    "Nutrients": ["氨氮_分指数", "总磷_分指数", "总氮_分指数"],
    "Toxic–metal": [
        "铜_分指数",
        "锌_分指数",
        "氟化物_分指数",
        "硒_分指数",
        "砷_分指数",
        "汞_分指数",
        "镉_分指数",
        "六价铬_分指数",
        "铅_分指数",
        "氰化物_分指数",
        "挥发酚_分指数",
        "石油类_分指数",
        "阴离子表面活性剂_分指数",
        "硫化物_分指数",
    ],
    "Microbial": ["粪大肠菌群_分指数"],
}


def fit_kmeans(x: np.ndarray, seed: int, n_init: int = 30) -> KMeans:
    return KMeans(
        n_clusters=N_STATES,
        n_init=n_init,
        random_state=seed,
    ).fit(x)


def order_clusters(raw_labels: np.ndarray, burden: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    summary = (
        pd.DataFrame({"cluster": raw_labels, "burden": burden})
        .groupby("cluster")["burden"]
        .median()
        .sort_values()
    )
    mapping = {int(cluster): rank for rank, cluster in enumerate(summary.index, start=1)}
    ordered = np.array([mapping[int(label)] for label in raw_labels], dtype=int)
    return ordered, mapping


def align_to_reference(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ref_levels = np.sort(np.unique(reference))
    pred_levels = np.sort(np.unique(predicted))
    overlap = np.zeros((len(ref_levels), len(pred_levels)), dtype=int)
    for i, ref in enumerate(ref_levels):
        for j, pred in enumerate(pred_levels):
            overlap[i, j] = np.sum((reference == ref) & (predicted == pred))
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {pred_levels[column]: ref_levels[row] for row, column in zip(rows, columns)}
    return np.array([mapping[value] for value in predicted], dtype=int)


def bootstrap_stability(x: np.ndarray, reference: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260730)
    global_rows: list[dict] = []
    state_rows: list[dict] = []
    n = len(x)
    for replicate in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, n)
        model = fit_kmeans(x[sample], seed=1000 + replicate, n_init=15)
        predicted = align_to_reference(reference, model.predict(x))
        global_rows.append(
            {
                "replicate": replicate + 1,
                "ARI": adjusted_rand_score(reference, predicted),
                "NMI": normalized_mutual_info_score(reference, predicted),
                "agreement": np.mean(reference == predicted),
            }
        )
        for state in range(1, N_STATES + 1):
            reference_set = reference == state
            predicted_set = predicted == state
            union = np.sum(reference_set | predicted_set)
            state_rows.append(
                {
                    "replicate": replicate + 1,
                    "risk_state": state,
                    "Jaccard": np.sum(reference_set & predicted_set) / union if union else np.nan,
                }
            )
    return pd.DataFrame(global_rows), pd.DataFrame(state_rows)


def group_profiles(scores: pd.DataFrame) -> pd.DataFrame:
    group_data = pd.DataFrame(index=scores.index)
    for group, columns in GROUPS.items():
        available = [column for column in columns if column in scores.columns]
        group_data[group] = scores[available].astype(float).mean(axis=1, skipna=True)

    medians = group_data.median(axis=0)
    mad = (group_data - medians).abs().median(axis=0)
    robust = (group_data - medians) / (1.4826 * mad.replace(0, np.nan))
    robust = robust.clip(-2.5, 2.5)
    robust["risk_state"] = scores["自然层级"].astype(int)
    profile = robust.groupby("risk_state").median().T
    profile.columns = [f"State {int(column)}" for column in profile.columns]
    return profile


def main() -> None:
    assignment = pd.read_csv(ASSIGNMENT_INPUT, encoding="utf-8-sig").copy()
    scores = pd.read_csv(SCORE_INPUT, encoding="utf-8-sig").copy()
    x = assignment[["PCA1", "PCA2"]].to_numpy(dtype=float)

    model = fit_kmeans(x, seed=SEED)
    raw_labels = model.labels_
    ordered_labels, ordering = order_clusters(
        raw_labels,
        assignment["污染指数型CWQI"].to_numpy(dtype=float),
    )

    assignment["最佳聚类方法"] = "KMeans"
    assignment["原始簇"] = raw_labels
    assignment["自然层级"] = ordered_labels
    assignment["自然层名称"] = [STATE_NAMES[level] for level in ordered_labels]
    assignment.to_csv(ASSIGNMENT_OUTPUT, index=False, encoding="utf-8-sig")

    key = assignment[["station_id", "time_id", "自然层级", "自然层名称"]]
    scores = scores.drop(columns=["自然层级", "自然层名称"], errors="ignore").merge(
        key,
        on=["station_id", "time_id"],
        how="left",
        validate="one_to_one",
    )
    scores.to_csv(SCORE_OUTPUT, index=False, encoding="utf-8-sig")

    global_stability, state_stability = bootstrap_stability(x, ordered_labels)
    global_stability.to_csv(
        VALIDATION_OUTPUT / "bootstrap_global_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    state_stability.to_csv(
        VALIDATION_OUTPUT / "bootstrap_state_jaccard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    profile = group_profiles(scores)
    profile.to_csv(
        VALIDATION_OUTPUT / "risk_state_group_profiles.csv",
        encoding="utf-8-sig",
    )

    state_summary = (
        assignment.groupby(["自然层级", "自然层名称"])
        .agg(
            samples=("自然层级", "size"),
            median_legacy_cwqi=("污染指数型CWQI", "median"),
            mean_legacy_cwqi=("污染指数型CWQI", "mean"),
        )
        .reset_index()
    )
    state_summary.to_csv(
        VALIDATION_OUTPUT / "kmeans_state_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = pd.DataFrame(
        [
            {
                "Method": "KMeans",
                "K": N_STATES,
                "Silhouette": silhouette_score(x, ordered_labels),
                "Calinski-Harabasz": calinski_harabasz_score(x, ordered_labels),
                "Davies-Bouldin": davies_bouldin_score(x, ordered_labels),
                "Bootstrap ARI median": global_stability["ARI"].median(),
                "Bootstrap ARI q025": global_stability["ARI"].quantile(0.025),
                "Bootstrap ARI q975": global_stability["ARI"].quantile(0.975),
            }
        ]
    )
    metrics.to_csv(
        VALIDATION_OUTPUT / "kmeans_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ordering_frame = pd.DataFrame(
        [{"raw_cluster": raw, "risk_state": state} for raw, state in ordering.items()]
    )
    ordering_frame.to_csv(
        VALIDATION_OUTPUT / "kmeans_cluster_ordering.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(state_summary.to_string(index=False))
    print("\n", metrics.to_string(index=False))
    print(f"\nAssignments written to: {ASSIGNMENT_OUTPUT}")
    print(f"Scores written to: {SCORE_OUTPUT}")


if __name__ == "__main__":
    main()

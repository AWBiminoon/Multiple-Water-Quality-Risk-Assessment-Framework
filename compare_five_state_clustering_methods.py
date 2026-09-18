"""Compare four clustering algorithms under a fixed five-state specification."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "viz_input" / "best_dimred_cluster_assignments.csv"
OUTPUT = ROOT / "cluster_validation_outputs" / "five_state_method_comparison.csv"

METHODS = ["KMeans", "Agglomerative", "Spectral", "GMM"]
N_STATES = 5
N_RESAMPLES = 50
SUBSAMPLE_FRACTION = 0.8


def fit_predict(method: str, x: np.ndarray, seed: int) -> np.ndarray:
    if method == "KMeans":
        return KMeans(
            n_clusters=N_STATES,
            n_init=30,
            random_state=seed,
        ).fit_predict(x)
    if method == "Agglomerative":
        return AgglomerativeClustering(
            n_clusters=N_STATES,
            linkage="ward",
        ).fit_predict(x)
    if method == "Spectral":
        return SpectralClustering(
            n_clusters=N_STATES,
            affinity="nearest_neighbors",
            n_neighbors=min(20, len(x) - 1),
            assign_labels="kmeans",
            n_init=20,
            random_state=seed,
        ).fit_predict(x)
    if method == "GMM":
        model = GaussianMixture(
            n_components=N_STATES,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=10,
            random_state=seed,
        )
        return model.fit_predict(x)
    raise ValueError(f"Unknown method: {method}")


def minmax(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    span = float(values.max() - values.min())
    scaled = pd.Series(1.0, index=values.index) if span == 0 else (values - values.min()) / span
    return scaled if higher_is_better else 1.0 - scaled


def main() -> None:
    assignment = pd.read_csv(INPUT, encoding="utf-8-sig")
    x = assignment[["PCA1", "PCA2"]].to_numpy(dtype=float)
    rng = np.random.default_rng(20260730)
    sample_size = int(round(len(x) * SUBSAMPLE_FRACTION))

    rows: list[dict[str, float | str | int]] = []
    for method in METHODS:
        reference = fit_predict(method, x, seed=42)
        stability = []
        for replicate in range(N_RESAMPLES):
            indices = np.sort(rng.choice(len(x), size=sample_size, replace=False))
            labels = fit_predict(method, x[indices], seed=1000 + replicate)
            stability.append(adjusted_rand_score(reference[indices], labels))

        rows.append(
            {
                "Method": method,
                "K": N_STATES,
                "Silhouette": silhouette_score(x, reference),
                "Calinski-Harabasz": calinski_harabasz_score(x, reference),
                "Davies-Bouldin": davies_bouldin_score(x, reference),
                "Resampling ARI median": float(np.median(stability)),
                "Resampling ARI q025": float(np.quantile(stability, 0.025)),
                "Resampling ARI q975": float(np.quantile(stability, 0.975)),
                "Resamples": N_RESAMPLES,
                "Subsample fraction": SUBSAMPLE_FRACTION,
            }
        )

    result = pd.DataFrame(rows)
    result["Silhouette support"] = minmax(result["Silhouette"], higher_is_better=True)
    result["CH support"] = minmax(result["Calinski-Harabasz"], higher_is_better=True)
    result["DB support"] = minmax(result["Davies-Bouldin"], higher_is_better=False)
    result["Stability support"] = minmax(
        result["Resampling ARI median"],
        higher_is_better=True,
    )
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()

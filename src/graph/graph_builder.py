from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_similarity_graph(
    frame: pd.DataFrame,
    feature_columns: list[str],
    k_neighbors: int = 8,
    max_nodes: int = 600,
    random_state: int = 42,
) -> dict[str, Any]:
    """Construct a transaction-similarity graph that does not use the target label.

    The underlying dataset has no real entity identifiers, so the graph reflects
    inferred similarity among transactions based on standardized feature vectors.
    """
    if frame.empty:
        return {"nodes": [], "edges": [], "message": "No transactions available for graph construction."}

    working = frame.copy()
    sample_size = min(max_nodes, len(working))
    sample = working.sample(n=sample_size, random_state=random_state)

    non_feature_columns = {
        "Class",
        "ground_truth_label",
        "transaction_id",
        "split",
        "model_probability",
        "model_prediction",
        "risk_category",
    }
    numeric_cols = [
        col
        for col in feature_columns
        if col in sample.columns
        and col not in non_feature_columns
        and pd.api.types.is_numeric_dtype(sample[col])
    ]
    if not numeric_cols:
        return {"nodes": [], "edges": [], "message": "No usable numeric feature columns available."}

    values = sample[numeric_cols].copy()
    values = values.fillna(values.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    neighbors = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(sample)), metric="euclidean")
    neighbors.fit(scaled)
    distances, indices = neighbors.kneighbors(scaled)

    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []

    for row_index, neighbor_info in enumerate(indices):
        source_id = sample.iloc[row_index]["transaction_id"]
        for neighbor_index, neighbor in enumerate(neighbor_info[1:]):
            neighbor_id = sample.iloc[neighbor]["transaction_id"]
            if source_id == neighbor_id:
                continue
            key = tuple(sorted((source_id, neighbor_id)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            distance = float(distances[row_index, neighbor_index + 1])
            similarity = float(1.0 / (1.0 + distance))
            edges.append(
                {
                    "source": source_id,
                    "target": neighbor_id,
                    "distance": distance,
                    "similarity": similarity,
                    "color": "rgba(94, 234, 212, 0.64)",
                }
            )

    if len(sample) > 1:
        projection = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
        sample["x"] = projection[:, 0]
        sample["y"] = projection[:, 1]
    else:
        sample["x"] = 0.0
        sample["y"] = 0.0

    nodes = []
    for _, row in sample.iterrows():
        nodes.append(
            {
                "id": str(row["transaction_id"]),
                "label": str(row["transaction_id"]),
                "risk_score": float(row.get("model_probability", 0.0)),
                "ground_truth": int(row.get("Class", -1)) if pd.notna(row.get("Class")) else None,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "split": str(row.get("split", "sample")),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "sample_size": len(sample),
        "k_neighbors": int(k_neighbors),
        "message": (
            "This graph connects transactions based on standardized feature similarity, "
            "not actual payment relationships."
        ),
    }

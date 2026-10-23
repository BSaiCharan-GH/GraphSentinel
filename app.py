from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.graph.graph_builder import build_similarity_graph
from src.data.artifact_store import (
    ArtifactStoreError,
    configuration_version,
    configure_artifact_store,
)
from src.models.fraud_model import (
    build_evaluation_frame,
    get_feature_contributions,
    load_trained_model,
)

LOGGER = logging.getLogger("graphsentinel.deployment")

def _risk_band(value: float) -> str:
    if value < 0.2:
        return "low"
    if value < 0.5:
        return "medium"
    if value < 0.8:
        return "high"
    return "critical"


def render_sidebar_signal(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    flagged = int((predictions["model_prediction"] == 1).sum())
    avg_score = float(predictions["model_probability"].mean())
    st.sidebar.markdown(
        f"""
        <div style="padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(148,163,184,0.22); background: rgba(15,23,42,0.8); margin-top: 14px;">
            <div style="font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: #94A3B8; margin-bottom: 8px;">Live queue</div>
            <div style="display: flex; justify-content: space-between; color: #E2E8F0; margin-bottom: 6px;">
                <span>Flagged</span>
                <strong>{flagged:,}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; color: #E2E8F0;">
                <span>Avg score</span>
                <strong>{avg_score:.3f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_prediction_table(config_version: int) -> pd.DataFrame:
    return build_evaluation_frame()


@st.cache_resource(show_spinner=False)
def get_model_artifact(config_version: int) -> dict:
    return load_trained_model(require_splits=True)


@st.cache_data(show_spinner=False)
def initialize_demo_data(config_version: int) -> str | None:
    try:
        get_model_artifact(config_version)
        get_prediction_table(config_version)
    except (
        ArtifactStoreError,
        FileNotFoundError,
        ValueError,
        TypeError,
        OSError,
        ImportError,
        pd.errors.ParserError,
        MemoryError,
    ) as error:
        LOGGER.exception("GraphSentinel demo initialization failed: %s", error)
        return (
            "The demo’s transaction analysis is temporarily unavailable. "
            "The Methodology & Data page remains available."
        )
    return None


@st.cache_data(show_spinner=False)
def get_similarity_graph(
    predictions: pd.DataFrame,
    feature_columns: tuple[str, ...],
    k_neighbors: int,
    max_nodes: int,
) -> dict:
    return build_similarity_graph(
        predictions,
        list(feature_columns),
        k_neighbors=k_neighbors,
        max_nodes=max_nodes,
    )


def render_kpi_card(label: str, value: str, accent: str = "#E2E8F0") -> None:
    st.markdown(
        f"""
        <div style="padding: 14px 16px; border:1px solid rgba(148,163,184,0.2); border-radius: 14px; background: rgba(15,23,42,0.82); margin-bottom: 10px;">
            <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">{label}</div>
            <div style="font-size: 1.7rem; font-weight: 700; color: {accent};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_setup_status() -> None:
    st.markdown(
        "Please try again later. The Methodology & Data page is available without "
        "transaction data."
    )


def show_artifact_error(error: Exception) -> None:
    LOGGER.exception("GraphSentinel page could not load transaction artifacts: %s", error)
    st.error("Transaction analysis is temporarily unavailable.")
    render_setup_status()


def render_overview() -> None:
    try:
        config_version = configuration_version()
        predictions = get_prediction_table(config_version)
        model_artifact = get_model_artifact(config_version)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        show_artifact_error(error)
        return

    summary = model_artifact.get("summary") or {}
    test_summary = summary.get("test", {})

    total_transactions = int(len(predictions))
    fraud_cases = int(predictions["Class"].sum())
    fraud_rate = (fraud_cases / total_transactions) if total_transactions else 0.0
    if predictions.empty:
        st.info("No transaction records are available in the configured artifacts.")
        return
    render_sidebar_signal(predictions)

    flagged_transactions = int((predictions["model_prediction"] == 1).sum())
    high_risk_transactions = int((predictions["model_probability"] >= 0.8).sum())
    precision = float(test_summary.get("precision", 0.0))
    recall = float(test_summary.get("recall", 0.0))
    f1_score = float(test_summary.get("f1", 0.0))
    pr_auc = float(test_summary.get("pr_auc", 0.0))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total transactions", f"{total_transactions:,}", "#E2E8F0")
    with col2:
        render_kpi_card("Fraud cases", f"{fraud_cases:,}", "#F97316")
    with col3:
        render_kpi_card("Model flagged", f"{flagged_transactions:,}", "#22D3EE")
    with col4:
        render_kpi_card("High-risk", f"{high_risk_transactions:,}", "#F43F5E")

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    metrics_col = st.columns(4)
    metric_values = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score,
        "PR-AUC": pr_auc,
    }
    for i, (name, metric_value) in enumerate(metric_values.items()):
        with metrics_col[i]:
            render_kpi_card(name, f"{metric_value:.3f}", "#67E8F9")

    left_col, right_col = st.columns([1.5, 1])
    with left_col:
        st.subheader("Volume and risk trends")
        plot_df = predictions.copy()
        plot_df["risk_band"] = plot_df["model_probability"].apply(_risk_band)
        plot_df["time_window"] = (plot_df["Time"] // 1000).astype(int)
        volume = (
            plot_df.groupby("time_window", as_index=False)
            .agg(transaction_count=("transaction_id", "count"), avg_risk=("model_probability", "mean"))
            .sort_values("time_window")
        )
        fig_volume = px.line(
            volume,
            x="time_window",
            y=["transaction_count", "avg_risk"],
            markers=True,
            title="Transaction volume vs. average model risk",
            color_discrete_sequence=["#22D3EE", "#F59E0B"],
        )
        fig_volume.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_volume, use_container_width=True)
    with right_col:
        st.subheader("System status")
        st.markdown(
            f"""
            <div style="padding: 16px 18px; border:1px solid rgba(148,163,184,0.2); border-radius: 14px; background: rgba(15,23,42,0.82); margin-bottom: 12px;">
                <div style="color: #67E8F9; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;">Dataset</div>
                <div style="font-size: 1.0rem; color: #E2E8F0;">{total_transactions:,} records</div>
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">Ground truth fraud rate: {fraud_rate:.4%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="padding: 16px 18px; border:1px solid rgba(148,163,184,0.2); border-radius: 14px; background: rgba(15,23,42,0.82); margin-bottom: 12px;">
                <div style="color: #67E8F9; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;">Alerting</div>
                <div style="font-size: 1.0rem; color: #E2E8F0;">Threshold {model_artifact.get('selected_threshold', 0.5):.4f}</div>
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">{flagged_transactions:,} transactions currently flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="padding: 16px 18px; border:1px solid rgba(148,163,184,0.2); border-radius: 14px; background: rgba(15,23,42,0.82);">
                <div style="color: #67E8F9; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;">Graph</div>
                <div style="font-size: 1.0rem; color: #E2E8F0;">Similarity network enabled</div>
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">Inferred analytical relationships only</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    risk_histogram = (
        plot_df.groupby("risk_band", dropna=False)
        .size()
        .reset_index(name="transactions")
    )
    risk_histogram["risk_band"] = risk_histogram["risk_band"].fillna("unknown")
    fig_risk = px.bar(
        risk_histogram,
        x="risk_band",
        y="transactions",
        color="risk_band",
        color_discrete_map={
            "low": "#22D3EE",
            "medium": "#F59E0B",
            "high": "#F97316",
            "critical": "#F43F5E",
            "unknown": "#94A3B8",
        },
        title="Risk distribution",
    )
    fig_risk.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_risk, use_container_width=True)

    flagged_recent = (
        predictions.loc[predictions["model_prediction"] == 1, ["transaction_id", "Time", "Amount", "model_probability", "Class"]]
        .sort_values("model_probability", ascending=False)
        .head(10)
        .copy()
    )
    flagged_recent["ground_truth"] = flagged_recent["Class"].map({0: "Legitimate", 1: "Fraud"})
    st.subheader("Recent flagged transactions")
    st.dataframe(flagged_recent.reset_index(drop=True), use_container_width=True, hide_index=True)


def render_feature_importance_chart(explanation: pd.DataFrame, top_n: int = 8) -> None:
    chart_df = explanation.head(top_n).copy()
    chart_df = chart_df.sort_values("contribution_score", ascending=True)
    chart_df["evidence"] = np.select(
        [chart_df["contribution"] > 0, chart_df["contribution"] < 0],
        ["Supports fraud", "Opposes fraud"],
        default="Neutral",
    )
    fig = px.bar(
        chart_df,
        x="contribution",
        y="feature",
        orientation="h",
        color="evidence",
        color_discrete_map={
            "Supports fraud": "#F87171",
            "Opposes fraud": "#38BDF8",
            "Neutral": "#94A3B8",
        },
        title="Signed contribution to fraud-class score",
        text="contribution",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="Evidence direction",
    )
    fig.add_vline(x=0, line_color="rgba(226,232,240,0.45)", line_width=1)
    fig.update_traces(
        texttemplate="%{text:+.3f}",
        textposition="outside",
        hovertemplate="%{y}<br>Signed logit contribution: %{x:+.4f}<extra></extra>",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_transaction_explorer() -> None:
    try:
        config_version = configuration_version()
        predictions = get_prediction_table(config_version)
        model_artifact = get_model_artifact(config_version)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        show_artifact_error(error)
        return

    if predictions.empty:
        st.info("Transaction data is not available yet. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]
    render_sidebar_signal(predictions)

    with st.sidebar:
        st.subheader("Transaction filters")
        risk_filter = st.selectbox("Predicted risk", ["All", "low", "medium", "high", "critical"])
        split_filter = st.multiselect("Split", options=["train", "validation", "test"], default=["train", "validation", "test"])
        amount_min = st.number_input("Minimum amount", min_value=0.0, value=0.0, step=10.0)
        time_min = int(predictions["Time"].min())
        time_max = int(predictions["Time"].max())
        time_window = st.slider("Time window", min_value=time_min, max_value=time_max, value=(time_min, time_max))
        search_term = st.text_input("Search by transaction ID")
        sort_by = st.selectbox("Sort by", ["Model score (high to low)", "Amount (high to low)", "Time (ascending)"])

    filtered = predictions.copy()
    filtered = filtered[filtered["split"].isin(split_filter)]
    filtered = filtered[filtered["Amount"] >= amount_min]
    filtered = filtered[(filtered["Time"] >= time_window[0]) & (filtered["Time"] <= time_window[1])]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_category"] == risk_filter]
    if search_term:
        filtered = filtered[filtered["transaction_id"].str.contains(search_term, case=False, na=False)]

    if sort_by == "Model score (high to low)":
        filtered = filtered.sort_values("model_probability", ascending=False)
    elif sort_by == "Amount (high to low)":
        filtered = filtered.sort_values("Amount", ascending=False)
    else:
        filtered = filtered.sort_values("Time", ascending=True)

    st.subheader("Filtered transactions")
    page_size = st.selectbox("Rows per page", [25, 50, 100], index=0, key="transaction_page_size")
    page_count = max(1, int(np.ceil(len(filtered) / page_size)))
    if int(st.session_state.get("transaction_page_number", 1)) > page_count:
        st.session_state["transaction_page_number"] = page_count
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key="transaction_page_number",
    )
    page_start = (int(page_number) - 1) * page_size
    visible_rows = filtered.iloc[page_start : page_start + page_size]
    st.caption(f"Showing {len(visible_rows):,} of {len(filtered):,} matching transactions")
    display_columns = [
        "transaction_id",
        "split",
        "Time",
        "Amount",
        "model_probability",
        "risk_category",
        "Class",
    ]
    st.dataframe(visible_rows[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Selected transaction")
    transaction_options = ["Select a transaction"] + visible_rows["transaction_id"].tolist()
    selected_id = st.selectbox("Choose a row", transaction_options)
    if selected_id == "Select a transaction":
        st.info("Choose a transaction to inspect its feature values and model interpretation.")
        return

    selected_row = filtered[filtered["transaction_id"] == selected_id].iloc[0]
    st.markdown(
        f"""
        <div style="padding: 14px 16px; border:1px solid rgba(148,163,184,0.2); border-radius: 12px; background: rgba(15,23,42,0.82);">
            <div style="font-size: 11px; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">Transaction details</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #E2E8F0;">{selected_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    score_value = float(selected_row["model_probability"])
    details = st.columns(4)
    details[0].metric("Model score", f"{score_value:.4f}")
    details[1].metric("Selected threshold", f"{model_artifact['selected_threshold']:.4f}")
    details[2].metric("Risk category", str(selected_row["risk_category"]))
    ground_truth = selected_row.get("Class")
    details[3].metric("Ground truth", "Fraud" if pd.notna(ground_truth) and int(ground_truth) == 1 else "Legitimate")

    explain = get_feature_contributions(
        model_artifact["model"],
        model_artifact["scaler"],
        selected_row,
        feature_columns,
    )
    st.subheader("Feature values and explanation")
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        explanation_cols = ["feature", "value", "coefficient", "contribution", "contribution_score"]
        st.dataframe(explain.head(12)[explanation_cols], use_container_width=True, hide_index=True)
    with right_col:
        render_feature_importance_chart(explain, top_n=8)


def render_graph_intelligence() -> None:
    try:
        config_version = configuration_version()
        predictions = get_prediction_table(config_version)
        model_artifact = get_model_artifact(config_version)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        show_artifact_error(error)
        return

    if predictions.empty:
        st.info("Graph construction depends on the evaluated transaction frame. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]
    render_sidebar_signal(predictions)

    with st.sidebar:
        st.subheader("Graph settings")
        k_neighbors = st.slider("k-nearest neighbors", min_value=3, max_value=20, value=8)
        max_nodes = st.slider("Sample size", min_value=100, max_value=min(len(predictions), 1000), value=min(600, len(predictions)))

    graph = get_similarity_graph(predictions, tuple(feature_columns), k_neighbors, max_nodes)
    if not graph.get("nodes") or not graph.get("edges"):
        st.warning("The similarity graph is unavailable with the current sample configuration. Try a different neighborhood size or a larger sample.")
        return

    avg_degree = (2 * len(graph["edges"])) / len(graph["nodes"]) if graph["nodes"] else 0.0
    flagged_share = float(sum(1 for node in graph["nodes"] if node["risk_score"] >= 0.8) / len(graph["nodes"])) if graph["nodes"] else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", f"{len(graph['nodes']):,}")
    col2.metric("Edges", f"{len(graph['edges']):,}")
    col3.metric("Avg degree", f"{avg_degree:.2f}")
    st.caption(f"Sampled flagged share: {flagged_share:.2%} of nodes exceed the high-risk threshold.")

    st.info(
        "Nodes represent transactions; edges connect similar transaction vectors by standardized feature similarity. "
        "These are inferred analytical relationships, not real payment customer-to-merchant networks."
    )
    node_lookup = {node["id"]: node for node in graph["nodes"]}
    edge_x, edge_y = [], []
    for edge in graph["edges"]:
        source = node_lookup.get(edge["source"])
        target = node_lookup.get(edge["target"])
        if source is None or target is None:
            continue
        edge_x.extend([source["x"], target["x"], None])
        edge_y.extend([source["y"], target["y"], None])

    node_x = [node["x"] for node in graph["nodes"]]
    node_y = [node["y"] for node in graph["nodes"]]
    node_risk = [node["risk_score"] for node in graph["nodes"]]
    node_text = [node["id"] for node in graph["nodes"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(94, 234, 212, 0.55)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            marker=dict(
                size=10,
                color=node_risk,
                colorscale=[[0, "#22D3EE"], [0.5, "#F59E0B"], [1, "#F43F5E"]],
                cmin=0,
                cmax=1,
                opacity=0.9,
                line=dict(color="rgba(255,255,255,0.5)", width=0.5),
            ),
            hovertemplate="Transaction: %{text}<br>Risk score: %{marker.color:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Transaction similarity graph",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_node_id = st.selectbox(
        "Inspect sampled transaction",
        options=list(node_lookup),
        key="graph_selected_node",
    )
    selected_node = node_lookup[selected_node_id]
    node_details = st.columns(3)
    node_details[0].metric("Selected transaction", selected_node_id)
    node_details[1].metric("Model score", f"{selected_node['risk_score']:.4f}")
    node_details[2].metric("Split", selected_node["split"])
    selected_edges = [
        {
            "neighbor_id": edge["target"] if edge["source"] == selected_node_id else edge["source"],
            "similarity": edge["similarity"],
            "distance": edge["distance"],
        }
        for edge in graph["edges"]
        if edge["source"] == selected_node_id or edge["target"] == selected_node_id
    ]
    if selected_edges:
        neighbors = pd.DataFrame(selected_edges).sort_values("similarity", ascending=False)
        st.dataframe(neighbors, use_container_width=True, hide_index=True)
    else:
        st.info("No sampled similarity neighbors are available for this transaction.")

    st.subheader("Graph construction notes")
    st.markdown(
        """
        - Nodes are transaction records sampled from the evaluated dataset.
        - Features are standardized to prevent scale differences from dominating neighborhood selection.
        - Similarity edges are built with a k-nearest-neighbor graph and avoid self-loops and duplicate links.
        - The target label is excluded from graph construction to keep the analysis independent of known ground truth.
        - Neighbor relationships are analytical approximations, not origin-to-destination payment paths.
        """
    )


def build_risk_review_payload(
    selected_id: str,
    selected_row: pd.Series,
    explanation: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    case_status: str,
    note: str,
    model_artifact: dict,
) -> dict:
    return {
        "transaction_id": selected_id,
        "split": str(selected_row.get("split", "unknown")),
        "time": float(selected_row.get("Time", 0.0)),
        "amount": float(selected_row.get("Amount", 0.0)),
        "model_probability": float(selected_row.get("model_probability", 0.0)),
        "risk_band": str(selected_row.get("risk_category", "unknown")),
        "selected_threshold": float(model_artifact.get("selected_threshold", 0.5)),
        "ground_truth": int(selected_row.get("Class", -1)) if pd.notna(selected_row.get("Class")) else None,
        "case_status": case_status,
        "investigation_note": note,
        "top_features": explanation.head(10).to_dict(orient="records"),
        "graph_neighbors": neighbor_df.to_dict(orient="records"),
    }


def render_fraud_investigation() -> None:
    try:
        config_version = configuration_version()
        predictions = get_prediction_table(config_version)
        model_artifact = get_model_artifact(config_version)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        show_artifact_error(error)
        return

    if predictions.empty:
        st.info("Fraud investigation requires evaluated predictions. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]
    render_sidebar_signal(predictions)
    flagged = predictions.loc[predictions["model_prediction"] == 1].sort_values("model_probability", ascending=False)

    if flagged.empty:
        st.warning("No flagged transactions are currently available for investigation.")
        return

    st.subheader("Flagged transaction review")
    selected_id = st.selectbox("Select a flagged transaction", flagged["transaction_id"].tolist())
    selected_row = flagged[flagged["transaction_id"] == selected_id].iloc[0]

    risk_summary = {
        "Model score": f"{float(selected_row['model_probability']):.4f}",
        "Risk band": str(selected_row["risk_category"]),
        "Threshold": f"{model_artifact['selected_threshold']:.4f}",
        "Ground truth": "Fraud" if pd.notna(selected_row.get("Class")) and int(selected_row.get("Class")) == 1 else "Legitimate",
    }
    for label, value in risk_summary.items():
        st.caption(f"{label}: {value}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model score", f"{float(selected_row['model_probability']):.4f}")
    col2.metric("Risk band", str(selected_row["risk_category"]))
    col3.metric("Threshold", f"{model_artifact['selected_threshold']:.4f}")
    truth = selected_row.get("Class")
    col4.metric("Ground truth", "Fraud" if pd.notna(truth) and int(truth) == 1 else "Legitimate")

    explanation = get_feature_contributions(
        model_artifact["model"],
        model_artifact["scaler"],
        selected_row,
        feature_columns,
    )
    top_features = explanation.head(5).copy()
    top_features["signal"] = np.select(
        [top_features["contribution"] > 0, top_features["contribution"] < 0],
        ["Supports fraud", "Opposes fraud"],
        default="Neutral",
    )
    st.markdown(
        """
        <div style="padding: 14px 16px; border:1px solid rgba(148,163,184,0.2); border-radius: 12px; background: rgba(15,23,42,0.82); margin-bottom: 12px;">
            <div style="font-size: 11px; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">Risk rationale</div>
            <div style="color: #E2E8F0; line-height: 1.6;">
            This alert is model-driven: the transaction has a high probability score relative to the selected operating threshold. The list below shows the largest feature contributions from the supervised model; these are model-based signals, not causal proof of fraud.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.dataframe(top_features[["feature", "value", "coefficient", "contribution", "signal"]], use_container_width=True, hide_index=True)
    with right_col:
        render_feature_importance_chart(explanation, top_n=6)

    graph = get_similarity_graph(predictions, tuple(feature_columns), 8, min(500, len(predictions)))
    neighbor_ids = []
    if graph.get("edges"):
        for edge in graph["edges"]:
            if edge["source"] == selected_id:
                neighbor_ids.append(edge["target"])
            elif edge["target"] == selected_id:
                neighbor_ids.append(edge["source"])
    if neighbor_ids:
        neighbor_df = predictions[predictions["transaction_id"].isin(neighbor_ids)][["transaction_id", "model_probability", "risk_category", "Class"]]
        st.subheader("Graph neighbors")
        st.dataframe(neighbor_df.sort_values("model_probability", ascending=False), use_container_width=True, hide_index=True)
    else:
        neighbor_df = pd.DataFrame(columns=["transaction_id", "model_probability", "risk_category", "Class"])
        st.caption("No nearby similarity neighbors were identified for this transaction.")

    status_options = ["Open", "Under review", "Escalated", "Closed"]
    note_default = st.session_state.get("investigation_note", "")
    case_status = st.selectbox("Case status", status_options, index=status_options.index(st.session_state.get("case_status", "Open")))
    st.session_state["case_status"] = case_status
    st.text_area("Local investigation note", value=note_default, key="investigation_note")

    review_payload = build_risk_review_payload(
        selected_id,
        selected_row,
        explanation,
        neighbor_df,
        case_status,
        st.session_state.get("investigation_note", ""),
        model_artifact,
    )
    export_json = json.dumps(review_payload, indent=2)
    st.download_button(
        label="Export risk review",
        data=export_json,
        file_name=f"risk_review_{selected_id}.json",
        mime="application/json",
    )


def render_model_lab() -> None:
    st.markdown(
        """
        <div class="page-intro">
            <div class="eyebrow">EVALUATION WORKSPACE</div>
            <div class="intro-copy">Inspect held-out model performance and the validation-selected operating threshold without mixing their roles.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        config_version = configuration_version()
        predictions = get_prediction_table(config_version)
        model_artifact = get_model_artifact(config_version)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        show_artifact_error(error)
        return

    if predictions.empty:
        st.info("No evaluated predictions are available. Train the baseline model to create evaluation results.")
        return

    test = predictions.loc[predictions["split"] == "test"].copy()
    validation = predictions.loc[predictions["split"] == "validation"].copy()
    if test.empty or validation.empty:
        st.warning("Both validation and held-out test predictions are required for Model Lab.")
        return

    threshold = float(model_artifact["selected_threshold"])
    render_sidebar_signal(predictions)
    y_test = test["Class"].astype(int).to_numpy()
    test_scores = test["model_probability"].to_numpy(dtype=float)
    test_predictions = (test_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        test_predictions,
        average="binary",
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)

    st.caption(
        f"Final evaluation uses the held-out test split. The operating threshold "
        f"({threshold:.6f}) was selected using validation data only."
    )
    metric_columns = st.columns(5)
    metrics = [
        ("Precision", precision),
        ("Recall", recall),
        ("F1-score", f1),
        ("ROC-AUC", roc_auc),
        ("PR-AUC", pr_auc),
    ]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            render_kpi_card(label, f"{value:.3f}", "#67E8F9")

    st.subheader("Held-out test evaluation")
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    cm_figure = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted legitimate", "Predicted fraud"],
            y=["Actual legitimate", "Actual fraud"],
            colorscale=[[0, "#0F172A"], [1, "#0891B2"]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "#F8FAFC"},
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    cm_figure.update_layout(
        title="Confusion matrix at validation-selected threshold",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Prediction",
        yaxis_title="Ground truth",
    )

    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_figure = go.Figure()
    roc_figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC {roc_auc:.3f}", line={"color": "#22D3EE", "width": 2}))
    roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"color": "#64748B", "dash": "dash"}))
    roc_figure.update_layout(
        title="ROC curve · held-out test",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.25),
    )

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_scores)
    pr_figure = go.Figure()
    pr_figure.add_trace(
        go.Scatter(
            x=recall_curve,
            y=precision_curve,
            mode="lines",
            name=f"PR-AUC {pr_auc:.3f}",
            line={"color": "#F59E0B", "width": 2},
        )
    )
    pr_figure.update_layout(
        title="Precision-recall curve · held-out test",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[0, 1.02]),
    )

    cm_col, roc_col = st.columns([1, 1.2])
    with cm_col:
        st.plotly_chart(cm_figure, use_container_width=True)
    with roc_col:
        st.plotly_chart(roc_figure, use_container_width=True)
    st.plotly_chart(pr_figure, use_container_width=True)

    st.subheader("Threshold trade-off · validation split")
    val_labels = validation["Class"].astype(int).to_numpy()
    val_scores = validation["model_probability"].to_numpy(dtype=float)
    val_precision, val_recall, val_thresholds = precision_recall_curve(val_labels, val_scores)
    if len(val_thresholds):
        sample_indexes = np.unique(
            np.linspace(0, len(val_thresholds) - 1, min(250, len(val_thresholds))).astype(int)
        )
        threshold_figure = go.Figure()
        threshold_figure.add_trace(
            go.Scatter(
                x=val_thresholds[sample_indexes],
                y=val_precision[:-1][sample_indexes],
                mode="lines",
                name="Precision",
                line={"color": "#22D3EE", "width": 2},
            )
        )
        threshold_figure.add_trace(
            go.Scatter(
                x=val_thresholds[sample_indexes],
                y=val_recall[:-1][sample_indexes],
                mode="lines",
                name="Recall",
                line={"color": "#F59E0B", "width": 2},
            )
        )
        threshold_figure.add_vline(
            x=threshold,
            line_color="#F87171",
            line_dash="dash",
            annotation_text="Selected threshold",
        )
        threshold_figure.update_layout(
            title="Validation precision and recall by decision threshold",
            xaxis_title="Decision threshold",
            yaxis_title="Metric",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(range=[0, 1.02]),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(threshold_figure, use_container_width=True)
    else:
        st.info("Threshold curve is unavailable because the validation split has no score thresholds.")

    st.caption(
        "Threshold is selected by maximizing validation F1 (ties prefer the higher threshold). "
        "Test curves and metrics are reported separately and are not used for threshold selection."
    )


def render_methodology_data() -> None:
    st.markdown(
        """
        <div class="page-intro">
            <div class="eyebrow">MODEL CARD · DATA LINEAGE</div>
            <div class="intro-copy">A transparent account of what GraphSentinel currently does—and what the source data cannot tell us.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Dataset and target")
        st.markdown(
            """
            The project uses the public ULB / Worldline credit-card fraud dataset. Each row is a transaction record with `Time`, `Amount`, anonymized PCA components `V1`–`V28`, and binary target `Class` (`0` legitimate, `1` fraud).

            The dataset contains no customer, card, merchant, or device identifiers. Its transaction IDs in the interface are generated from split and row position; they are not payment-provider IDs.
            """
        )
        st.subheader("Validation and preprocessing")
        st.markdown(
            """
            `src/data/validate_dataset.py` checks the expected columns, numeric/finite values, missing values, non-negative time and amount, and binary labels. `src/data/split_dataset.py` creates a reproducible, stratified 70% train / 15% validation / 15% test split using seed `42`.

            The model pipeline fits `StandardScaler` on training features only, then applies that scaler to validation and test features. The binary target is excluded from model inputs.
            """
        )
    with right:
        st.subheader("Baseline and evaluation")
        st.markdown(
            """
            The baseline is scikit-learn logistic regression with `class_weight=\"balanced\"`, `random_state=42`, and `max_iter=2000`. Precision, recall, F1, ROC-AUC, and average precision (PR-AUC) describe performance; confusion matrix and curves are calculated from actual split predictions.

            The decision threshold is selected by maximizing F1 on validation predictions; ties prefer the higher threshold. The held-out test split is reserved for final reporting and does not select the threshold.
            """
        )
        st.subheader("Signed feature contributions")
        st.markdown(
            """
            For the binary linear model, each feature contribution is its standardized value multiplied by the coefficient oriented toward fraud class `1`. Positive values support the fraud-class logit; negative values oppose it. Features are ranked by absolute contribution magnitude while retaining the sign.

            These are additive model signals in logit space—not causal explanations and not direct changes in probability.
            """
        )

    st.subheader("Transaction-similarity graph")
    st.markdown(
        """
        The graph standardizes the explicitly supplied model feature columns, samples a bounded number of transactions, and connects each transaction to its nearest neighbors using Euclidean distance. PCA projects the sample to two dimensions for visualization. `Class` and prediction metadata are excluded from distance calculations.

        **Edges represent inferred similarity between transaction feature vectors only. They are not real customer, card, merchant, device, or payment-network relationships.**
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - PCA-transformed inputs limit domain-level interpretation of `V1`–`V28`.
        - Fraud is highly imbalanced; metrics depend on the selected operating threshold and evaluation split.
        - A stratified random split is a baseline protocol, not a substitute for time-based or production validation.
        - Similarity edges are analytical neighbors, not evidence of collusion or shared ownership.
        - This project is a local prototype and does not connect to Razorpay live infrastructure.
        """
    )


def main() -> None:
    st.set_page_config(page_title="GraphSentinel | Fraud Intelligence", page_icon="🛡️", layout="wide")
    configure_artifact_store()
    st.markdown(
        """
        <style>
            :root { color-scheme: dark; }
            .stApp { background: #070D18; color: #E6EDF7; }
            [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 18% 0%, rgba(19, 47, 78, 0.22), transparent 36%), #070D18; }
            [data-testid="stHeader"] { background: rgba(7, 13, 24, 0.82); }
            [data-testid="stMainBlockContainer"] { max-width: 1500px; padding: 2rem clamp(1rem, 3vw, 2.75rem) 3rem; }
            div[data-testid="stSidebar"] { background: #0A1220; border-right: 1px solid rgba(148,163,184,0.13); }
            section[data-testid="stSidebarContent"] { padding: 1.15rem 0.85rem; }
            h1, h2, h3 { color: #F1F5FB !important; letter-spacing: -0.025em; }
            h1 { font-size: clamp(1.75rem, 3vw, 2.4rem) !important; margin: 0.25rem 0 1.2rem !important; }
            h2 { font-size: 1.25rem !important; margin-top: 1.7rem !important; }
            h3 { font-size: 1rem !important; }
            p, li, label, [data-testid="stCaptionContainer"] { color: #A8B5C8; }
            .page-intro { border: 1px solid rgba(148,163,184,0.15); border-radius: 16px; background: linear-gradient(115deg, rgba(15,27,44,0.92), rgba(11,20,33,0.84)); padding: 1.15rem 1.35rem; margin: 0 0 1.25rem; }
            .eyebrow { color: #67DDF2; font-size: 0.68rem; font-weight: 700; letter-spacing: .16em; margin-bottom: .45rem; }
            .intro-copy { color: #CFD9E7; font-size: 1rem; line-height: 1.55; max-width: 72rem; }
            div[data-testid="stMetric"] { border: 1px solid rgba(148,163,184,0.15); border-radius: 14px; background: #0D1726; padding: .85rem 1rem; }
            div[data-testid="stMetricLabel"] { color: #98A8BC; }
            div[data-testid="stMetricValue"] { color: #ECF6FF; font-weight: 650; }
            div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid rgba(148,163,184,0.16); border-radius: 12px; overflow: hidden; }
            div[data-testid="stPlotlyChart"] { border: 1px solid rgba(148,163,184,0.12); border-radius: 14px; background: rgba(13,23,38,0.48); padding: .35rem; }
            [data-testid="stVerticalBlockBorderWrapper"] { border-color: rgba(148,163,184,0.15) !important; border-radius: 14px !important; }
            .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid rgba(56,189,248,0.35); background: #102238; color: #E6F8FF; transition: background .16s ease, border-color .16s ease, transform .16s ease; }
            .stButton > button:hover, .stDownloadButton > button:hover { background: #16314D; border-color: #38BDF8; transform: translateY(-1px); }
            .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, input:focus-visible { outline: 2px solid #38BDF8 !important; outline-offset: 2px; }
            [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background: #0C1726; border-color: rgba(148,163,184,0.22); border-radius: 10px; }
            [data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover, [data-baseweb="textarea"] > div:hover { border-color: rgba(56,189,248,0.58); }
            div[role="radiogroup"] label { border: 1px solid transparent; border-radius: 10px; padding: .35rem .5rem; transition: background .15s ease, border-color .15s ease; }
            div[role="radiogroup"] label:hover { background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.2); }
            .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom-color: rgba(148,163,184,0.16); }
            .stTabs [data-baseweb="tab"] { border-radius: 9px 9px 0 0; color: #A8B5C8; }
            .stTabs [aria-selected="true"] { color: #E6F8FF !important; border-bottom-color: #38BDF8 !important; }
            @media (max-width: 760px) {
                [data-testid="stMainBlockContainer"] { padding: 1rem .85rem 2rem; }
                .page-intro { padding: .9rem 1rem; }
                div[data-testid="stMetric"] { padding: .7rem .75rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="padding: 18px 16px; border: 1px solid rgba(148,163,184,0.18); border-radius: 15px; background: #101D2E; margin-bottom: 12px;">
            <div style="font-size: 11px; letter-spacing: 0.18em; color: #67E8F9; text-transform: uppercase; margin-bottom: 10px;">Fraud intelligence</div>
            <div style="font-size: 2rem; font-weight: 700; color: #F8FAFC; line-height: 1.1;">GraphSentinel</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">Explainable payment-risk intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Portfolio-ready transaction risk workspace")

    nav = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Explorer",
            "Graph Intelligence",
            "Fraud Investigation",
            "Model Lab",
            "Methodology & Data",
        ],
        index=0,
        key="main_navigation",
    )

    with st.spinner("Preparing the transaction risk demo…"):
        initialization_error = initialize_demo_data(configuration_version())

    if initialization_error:
        st.warning(initialization_error)
        if nav == "Methodology & Data":
            st.title("Methodology & Data")
            render_methodology_data()
        else:
            st.info(
                "No transaction rows, scores, metrics, or graph relationships are being "
                "shown until the demo data is available. Select **Methodology & Data** "
                "in the sidebar to explore the implementation."
            )
        return

    if nav == "Overview":
        st.title("Overview")
        render_overview()
    elif nav == "Transaction Explorer":
        st.title("Transaction Explorer")
        render_transaction_explorer()
    elif nav == "Graph Intelligence":
        st.title("Graph Intelligence")
        render_graph_intelligence()
    elif nav == "Fraud Investigation":
        st.title("Fraud Investigation")
        render_fraud_investigation()
    elif nav == "Model Lab":
        st.title("Model Lab")
        render_model_lab()
    else:
        st.title("Methodology & Data")
        render_methodology_data()


if __name__ == "__main__":
    main()

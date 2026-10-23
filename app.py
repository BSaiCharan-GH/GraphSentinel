from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.graph.graph_builder import build_similarity_graph
from src.models.fraud_model import (
    build_evaluation_frame,
    get_feature_contributions,
    load_split_data,
    load_trained_model,
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_ROOT / "processed"


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


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_prediction_table() -> pd.DataFrame:
    try:
        return build_evaluation_frame()
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()


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
    processed_dir = ROOT / "data" / "processed"
    dataset_ready = RAW_DATA_PATH.exists()
    model_ready = (processed_dir / "fraud_model.joblib").exists()
    summary_ready = (processed_dir / "evaluation_summary.json").exists()

    if not dataset_ready or not model_ready:
        with st.container():
            st.warning("Repository setup required before review workflows can run.")
            st.code("python src/data/validate_dataset.py\npython src/data/split_dataset.py\npython src/models/train_baseline.py")
            st.markdown(
                """
                <div style="padding: 14px 16px; border:1px solid rgba(148,163,184,0.2); border-radius: 12px; background: rgba(15,23,42,0.82);">
                    <div style="font-size: 11px; color: #94A3B8; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">Setup checklist</div>
                    <div style="color: #E2E8F0; line-height: 1.7;">
                        • Dataset present: %s<br>
                        • Model artifacts present: %s<br>
                        • Held-out evaluation summary: %s
                    </div>
                </div>
                """ % (
                    "Yes" if dataset_ready else "No",
                    "Yes" if model_ready else "No",
                    "Yes" if summary_ready else "No",
                ),
                unsafe_allow_html=True,
            )


def render_overview() -> None:
    try:
        raw_df = load_raw_dataset()
    except FileNotFoundError:
        render_setup_status()
        st.warning("The raw dataset is not available yet. Place the credit-card CSV in data/raw/creditcard.csv and rerun the validation step.")
        return

    try:
        predictions = get_prediction_table()
        model_artifact = load_trained_model()
    except (FileNotFoundError, ValueError):
        render_setup_status()
        st.info("No trained model output is available yet. Run the baseline training step to populate evaluation metrics.")
        return

    summary = model_artifact.get("summary") or {}
    test_summary = summary.get("test", {})

    total_transactions = int(len(raw_df))
    fraud_cases = int(raw_df["Class"].sum())
    fraud_rate = (fraud_cases / total_transactions) if total_transactions else 0.0

    if predictions.empty:
        render_setup_status()
        st.info("No trained model output is available yet. Run the baseline training step to populate evaluation metrics.")
        return

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
    chart_df = chart_df.sort_values("contribution_score", ascending=False)
    chart_df["direction"] = np.where(chart_df["coefficient"] >= 0, "positive", "negative")
    fig = px.bar(
        chart_df,
        x="contribution_score",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"positive": "#22D3EE", "negative": "#F59E0B"},
        title="Top model-driven feature contributions",
        text="contribution_score",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def render_transaction_explorer() -> None:
    try:
        predictions = get_prediction_table()
        model_artifact = load_trained_model()
    except (FileNotFoundError, ValueError):
        st.info("Transaction data is not available yet. Train the baseline model first.")
        return

    if predictions.empty:
        st.info("Transaction data is not available yet. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]

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
    display_columns = [
        "transaction_id",
        "split",
        "Time",
        "Amount",
        "model_probability",
        "risk_category",
        "Class",
    ]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Selected transaction")
    transaction_options = ["Select a transaction"] + filtered["transaction_id"].tolist()
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
        explanation_cols = ["feature", "value", "coefficient", "contribution_score"]
        st.dataframe(explain.head(12)[explanation_cols], use_container_width=True, hide_index=True)
    with right_col:
        render_feature_importance_chart(explain, top_n=8)


def render_graph_intelligence() -> None:
    try:
        predictions = get_prediction_table()
        model_artifact = load_trained_model()
    except (FileNotFoundError, ValueError):
        st.info("Graph construction depends on the evaluated transaction frame. Train the baseline model first.")
        return

    if predictions.empty:
        st.info("Graph construction depends on the evaluated transaction frame. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]

    with st.sidebar:
        st.subheader("Graph settings")
        k_neighbors = st.slider("k-nearest neighbors", min_value=3, max_value=20, value=8)
        max_nodes = st.slider("Sample size", min_value=100, max_value=min(len(predictions), 1000), value=min(600, len(predictions)))

    graph = build_similarity_graph(predictions, feature_columns, k_neighbors=k_neighbors, max_nodes=max_nodes)
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
            mode="markers+text",
            text=node_text,
            textposition="top center",
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
        predictions = get_prediction_table()
        model_artifact = load_trained_model()
    except (FileNotFoundError, ValueError):
        st.info("Fraud investigation requires evaluated predictions. Train the baseline model first.")
        return

    if predictions.empty:
        st.info("Fraud investigation requires evaluated predictions. Train the baseline model first.")
        return

    feature_columns = model_artifact["feature_columns"]
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
    top_features["signal"] = np.where(top_features["coefficient"] >= 0, "Positive evidence", "Negative evidence")
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
        st.dataframe(top_features[["feature", "value", "coefficient", "contribution_score", "signal"]], use_container_width=True, hide_index=True)
    with right_col:
        render_feature_importance_chart(explanation, top_n=6)

    graph = build_similarity_graph(predictions, feature_columns, k_neighbors=8, max_nodes=min(500, len(predictions)))
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


def main() -> None:
    st.set_page_config(page_title="GraphSentinel | Fraud Intelligence", page_icon="🛡️", layout="wide")
    st.markdown(
        """
        <style>
            .stApp { background: #020817; color: #E2E8F0; }
            div[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.97); }
            div[data-testid="stSidebarNav"] { background: rgba(15, 23, 42, 0.95); }
            section[data-testid="stSidebarContent"] { padding: 1rem 0.8rem; }
            .stTabs [data-baseweb="tab-list"] { gap: 12px; }
            .stTabs [data-baseweb="tab"] { border-radius: 10px; color: #CBD5E1; }
            .stTabs [data-baseweb="tab-selected"] { background: rgba(34, 211, 238, 0.12); border: 1px solid rgba(34, 211, 238, 0.4); color: #E2E8F0; }
            .stDataFrame { background: rgba(15, 23, 42, 0.75); }
            .css-1d391kg, .css-18e3th9 { background: rgba(15, 23, 42, 0.0); }
            .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div, .stNumberInput > div > div { background: rgba(15,23,42,0.75); border: 1px solid rgba(148,163,184,0.2); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="padding: 18px 16px; border: 1px solid rgba(148,163,184,0.25); border-radius: 16px; background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(34,211,238,0.08)); margin-bottom: 12px; box-shadow: 0 10px 30px rgba(15,23,42,0.25);">
            <div style="font-size: 11px; letter-spacing: 0.18em; color: #67E8F9; text-transform: uppercase; margin-bottom: 10px;">Fraud intelligence</div>
            <div style="font-size: 2rem; font-weight: 700; color: #F8FAFC; line-height: 1.1;">GraphSentinel</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">Explainable payment-risk intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Portfolio-ready transaction risk workspace")

    predictions = get_prediction_table()
    render_sidebar_signal(predictions)

    nav = st.sidebar.radio(
        "Navigation",
        ["Overview", "Transaction Explorer", "Graph Intelligence", "Fraud Investigation"],
        index=0,
    )

    if nav == "Overview":
        st.title("Overview")
        render_overview()
    elif nav == "Transaction Explorer":
        st.title("Transaction Explorer")
        render_transaction_explorer()
    elif nav == "Graph Intelligence":
        st.title("Graph Intelligence")
        render_graph_intelligence()
    else:
        st.title("Fraud Investigation")
        render_fraud_investigation()


if __name__ == "__main__":
    main()

"""
Model Comparison — three tabs comparing the five trained models.
Pulls metrics directly from artifacts/metadata.json (no re-training).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_metadata,
    apply_plotly_theme, inject_global_css,
    PLUM, SAGE, SAGE_DARK, ACCENT_ROSE, CREAM, INK,
    COLORWAY,
)
st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
inject_global_css()

st.markdown(f"<h1 style='color:{PLUM};'>⚖️ Model Comparison</h1>",
            unsafe_allow_html=True)
st.markdown("*Five models, same data, same chance-level result.*")
st.markdown("---")

meta = load_metadata()
metrics = meta["metrics"]
best_model_name = meta["best_model"]

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Per-Model Deep Dive",
                            "🎲 Cross-Validation"])


# =========================================================================
# TAB 1 — OVERVIEW
# =========================================================================
with tab1:
    # KPI strip at top
    k1, k2, k3, k4 = st.columns(4)
    accuracies = [m["accuracy"] for m in metrics.values()]
    k1.metric("Models compared", len(metrics))
    k2.metric("Best test accuracy",
              f"{max(accuracies):.1%}",
              help=f"Achieved by {best_model_name}")
    k3.metric("Worst test accuracy", f"{min(accuracies):.1%}")
    k4.metric("Spread (max − min)",
              f"{(max(accuracies) - min(accuracies))*100:.2f} pp",
              help="All models within ~1 percentage point of each other.")

    st.markdown("---")

    # Accuracy bar chart with the chance line at 50%
    st.markdown("### Test-Set Accuracy by Model")
    acc_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "Accuracy": [m["accuracy"] for m in metrics.values()],
    }).sort_values("Accuracy", ascending=True)

    fig = px.bar(
        acc_df, x="Accuracy", y="Model", orientation="h",
        text=[f"{v:.2%}" for v in acc_df["Accuracy"]],
        color_discrete_sequence=[PLUM],
    )
    fig.update_traces(textposition="outside")
    fig.add_vline(
        x=0.5, line_dash="dash", line_color=ACCENT_ROSE, line_width=2,
        annotation_text="Chance level (50%)",
        annotation_position="top",
        annotation=dict(font=dict(color=ACCENT_ROSE, size=12)),
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis_title="", xaxis_title="Test-set accuracy",
        height=380,
    )
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    # Combined metrics table
    st.markdown("### All Metrics Side by Side")
    table_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "Accuracy": [m["accuracy"] for m in metrics.values()],
        "Precision": [m["precision"] for m in metrics.values()],
        "Recall": [m["recall"] for m in metrics.values()],
        "F1": [m["f1"] for m in metrics.values()],
        "CV mean": [m["cv_mean"] for m in metrics.values()],
        "CV std": [m["cv_std"] for m in metrics.values()],
    }).set_index("Model")

    st.dataframe(
        table_df.style
            .format({c: "{:.3f}" for c in table_df.columns})
            .background_gradient(subset=["Accuracy", "F1"], cmap="Purples"),
        use_container_width=True,
    )

    st.warning(
        f"""
        **The headline finding:** Across five fundamentally different
        algorithms — a linear model (Logistic Regression), a non-parametric
        method (KNN), a tree-based method (Decision Tree), an ensemble
        (Random Forest), and a kernel method (SVM) — every single one
        converged to within ~1 percentage point of 50%.

        When models with completely different inductive biases all agree
        that they cannot do better than chance, the bottleneck is **not
        the model**. It's the data. We prove this on the Insights page using
        mutual information.
        """
    )


# =========================================================================
# TAB 2 — PER-MODEL DEEP DIVE
# =========================================================================
with tab2:
    st.markdown("### Pick a model to inspect")

    chosen = st.selectbox(
        "Model",
        options=list(metrics.keys()),
        index=list(metrics.keys()).index(best_model_name),
    )
    m = metrics[chosen]

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1 Score", f"{m['f1']:.3f}")

    st.markdown("---")

    cm_col, hp_col = st.columns([1.2, 1])

# Confusion matrix
    with cm_col:
        st.markdown("#### Confusion Matrix")
        cm = np.array(m["confusion_matrix"])

        # Auto-pick text colour per cell so it stays readable on dark vs light background
        cm_max = cm.max()
        annotations = []
        for i in range(2):
            row = []
            for j in range(2):
                txt_color = "#FFFFFF" if cm[i, j] > 0.5 * cm_max else INK
                row.append(
                    f"<span style='font-size:32px; color:{txt_color};'><b>{cm[i, j]:,}</b></span>"
                )
            annotations.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: Failure", "Predicted: Success"],
            y=["Actual: Failure", "Actual: Success"],
            text=annotations,
            texttemplate="%{text}",
            colorscale=[[0, CREAM], [1, PLUM]],
            showscale=False,
            hoverinfo="skip",
        ))
        # Theme manually — apply_plotly_theme conflicts with heatmap title rendering
        fig.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor=CREAM,
            plot_bgcolor=CREAM,
            font=dict(family="Georgia, serif", color=INK, size=13),
            xaxis=dict(side="bottom", tickfont=dict(size=13, color=INK)),
            yaxis=dict(tickfont=dict(size=13, color=INK)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Hyperparameters + interpretation
    with hp_col:
        st.markdown("#### Best Hyperparameters")
        st.markdown(
            f"<div style='background:{CREAM}; padding:1rem; "
            f"border-radius:10px; border-left:4px solid {PLUM};'>",
            unsafe_allow_html=True,
        )
        for param, value in m["best_params"].items():
            st.markdown(f"- **`{param}`** = `{value}`")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Cross-Validation")
        st.markdown(
            f"- **5-fold CV mean accuracy:** `{m['cv_mean']:.4f}`  \n"
            f"- **CV standard deviation:** `{m['cv_std']:.4f}`  \n"
            f"- **CV scores per fold:** "
            f"`{[round(s, 4) for s in m['cv_scores']]}`"
        )

    st.markdown("---")
    st.markdown("#### How to read the confusion matrix")
    st.markdown(
        """
        - **TN (top-left)** — true negatives: model correctly said "Failure"
        - **FP (top-right)** — false positives: model said "Success" but it was actually a failure
        - **FN (bottom-left)** — false negatives: model said "Failure" but it was actually a success
        - **TP (bottom-right)** — true positives: model correctly said "Success"

        On a balanced binary task with no signal, a model behaves like a fair
        coin: TN ≈ TP and FP ≈ FN. That's roughly the pattern you see for every
        model here.
        """
    )


# =========================================================================
# TAB 3 — CROSS-VALIDATION
# =========================================================================
with tab3:
    st.markdown("### 5-Fold Cross-Validation")
    st.markdown(
        "*Cross-validation tells us if the chance-level performance is "
        "stable across different train/test splits, or just bad luck.*"
    )

    cv_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "CV Mean": [m["cv_mean"] for m in metrics.values()],
        "CV Std": [m["cv_std"] for m in metrics.values()],
    })

    # Bar chart with error bars
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cv_df["Model"],
        y=cv_df["CV Mean"],
        error_y=dict(type="data", array=cv_df["CV Std"], color=ACCENT_ROSE,
                     thickness=2, width=8),
        marker_color=PLUM,
        text=[f"{v:.3f}" for v in cv_df["CV Mean"]],
        textposition="outside",
    ))
    fig.add_hline(
        y=0.5, line_dash="dash", line_color=ACCENT_ROSE,
        annotation_text="Chance level (50%)",
        annotation_position="top right",
        annotation=dict(font=dict(color=ACCENT_ROSE)),
    )
    fig.update_layout(
        title="CV Mean Accuracy ± 1 Standard Deviation",
        yaxis=dict(range=[0.4, 0.6], tickformat=".0%"),
        xaxis_title="", yaxis_title="Cross-validation accuracy",
        height=420,
    )
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    # Per-fold scores grid
    st.markdown("### Per-Fold Scores")
    fold_data = []
    for name, m in metrics.items():
        for i, score in enumerate(m["cv_scores"]):
            fold_data.append({"Model": name, "Fold": f"Fold {i+1}",
                              "Accuracy": score})
    fold_df = pd.DataFrame(fold_data)

    fig = px.line(
    	fold_df, x="Fold", y="Accuracy", color="Model",
    	markers=True,
       	title="Accuracy across the 5 folds, per model",
        	color_discrete_sequence=COLORWAY,
    )

    fig.update_traces(line=dict(width=2.5), marker=dict(size=10))
    fig.add_hline(
        y=0.5, line_dash="dash", line_color=ACCENT_ROSE,
        annotation_text="Chance",
        annotation_position="bottom right",
        annotation=dict(font=dict(color=ACCENT_ROSE)),
    )
    fig.update_layout(yaxis=dict(range=[0.45, 0.55], tickformat=".0%"),
                      height=420)
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.info(
        """
        **What you're seeing:**

        Every fold of every model lands within roughly ±1 percentage point
        of 50%. The error bars (≈0.005 = half a percentage point) tell us
        the variation across folds is *tiny*. This isn't a fluke train/test
        split — the models genuinely cannot do better than chance on this
        data, no matter how the data is sliced.

        Compare this to a problem with real signal: cross-validation scores
        would cluster well above chance with similarly tight bounds. Here,
        they cluster *at* chance with similarly tight bounds. That's a
        fundamental result, not a tuning problem.
        """
    )
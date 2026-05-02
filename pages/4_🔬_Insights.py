"""
Insights — the conceptual capstone. Mutual information, PCA, and the verdict.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_mutual_information, load_pca, load_metadata,
    apply_plotly_theme, inject_global_css,
    PLUM, SAGE, SAGE_DARK, ACCENT_ROSE, CREAM, WARM_WHITE, INK,
)

st.set_page_config(page_title="Insights", page_icon="🔬", layout="wide")
inject_global_css()

st.markdown(f"<h1 style='color:{PLUM};'>🔬 Insights</h1>", unsafe_allow_html=True)
st.markdown("*Three angles on the same finding: there's no signal in the data.*")
st.markdown("---")

mi_df = load_mutual_information()
pca, X_pca, y_pca = load_pca()
meta = load_metadata()

# Build the organic-features view once, shared across tabs
interest_cols = set(meta["interest_columns"])
organic_mi = mi_df[~mi_df["feature"].isin(interest_cols)] \
                .reset_index(drop=True)

tab1, tab2, tab3 = st.tabs(["📈 Mutual Information",
                            "🌌 PCA Visualisation",
                            "🧭 The Verdict"])

# =========================================================================
# TAB 1 — MUTUAL INFORMATION
# =========================================================================
with tab1:
    st.markdown("### How much does each feature actually tell us about the target?")
    st.markdown(
        """
        **Mutual Information (MI)** measures, in bits, how much knowing the
        value of a feature reduces our uncertainty about the target. A
        perfectly predictive feature would score around 1 bit. A useless
        feature scores 0.

        We exclude the 49 individual hobby one-hots from this view (each
        one is mostly noise on its own), but we **keep the three engineered
        binary flags** — `feat_income_high`, `feat_edu_high`, and
        `feat_intent_high` — because they're useful summaries of the
        underlying text columns and are worth comparing directly against
        the organic demographic and behavioural features.
        """
    )

    st.caption(
        f"Showing **{len(organic_mi)}** features "
        f"(filtered out {len(interest_cols)} hobby tags)."
    )

    top_n = st.slider("How many top features to show?",
                      5, len(organic_mi), min(10, len(organic_mi)))

    top_df = organic_mi.head(top_n).sort_values("mi_score", ascending=True)
    fig = px.bar(
        top_df, x="mi_score", y="feature", orientation="h",
        text=[f"{v:.4f}" for v in top_df["mi_score"]],
        color_discrete_sequence=[PLUM],
    )
    fig.update_traces(textposition="outside")
    fig.add_vline(
        x=0.01, line_dash="dot", line_color=ACCENT_ROSE,
        annotation_text="0.01 bits — effectively noise",
        annotation_position="top",
        annotation=dict(font=dict(color=ACCENT_ROSE, size=11)),
    )
    fig.update_layout(
        xaxis_title="Mutual information (bits)",
        yaxis_title="",
        height=max(420, top_n * 22),
        xaxis=dict(range=[0, max(0.015, top_df["mi_score"].max() * 1.3)]),
        paper_bgcolor=WARM_WHITE,
        plot_bgcolor=WARM_WHITE,
        font=dict(family="Georgia, serif", color=INK, size=13),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
# Headline numbers (computed on organic features only)
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest MI score (organic)",
              f"{organic_mi['mi_score'].max():.4f}",
              help=f"Feature: {organic_mi.iloc[0]['feature']}")
    c2.metric("Mean MI across organic features",
              f"{organic_mi['mi_score'].mean():.4f}")
    c3.metric("Organic features above 0.01 bits",
              int((organic_mi["mi_score"] > 0.01).sum()),
              help=f"Out of {len(organic_mi)} organic features.")

st.warning(
        f"""
        **What this means in plain English:**

        Our **best organic feature** ({organic_mi.iloc[0]['feature']})
        carries just **{organic_mi.iloc[0]['mi_score']:.4f} bits** of
        information about the target — less than one-hundredth of a bit.
        The mean across all {len(organic_mi)} organic features is
        **{organic_mi['mi_score'].mean():.4f} bits**.

        For reference, a feature with genuine predictive power on a binary
        target would typically score 0.1–0.5 bits. Every demographic and
        behavioural feature we have falls **more than 10× short** of that
        threshold.

        This is why all five models converged to chance: there was nothing
        meaningful to learn from the features that actually represent user
        behaviour and identity.
        """
    )


# =========================================================================
# TAB 2 — PCA VISUALISATION
# =========================================================================
with tab2:
    st.markdown("### What does the data 'look like' in 2D?")
    st.markdown(
        """
        **Principal Component Analysis (PCA)** projects our 72-dimensional
        feature space down to 2 dimensions while preserving as much
        variance as possible. If Success and Failure profiles were
        meaningfully different, we'd expect to see two distinguishable
        clouds. We don't.
        """
    )

    # Subsample for performance
    sample_size = min(3000, len(X_pca))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_pca), sample_size, replace=False)

    pca_df = pd.DataFrame({
        "PC1": X_pca[idx, 0],
        "PC2": X_pca[idx, 1],
        "Outcome": ["Success" if y == 1 else "Failure" for y in y_pca[idx]],
    })

    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Outcome",
        opacity=0.45,
        title=f"PCA projection of {sample_size:,} balanced training profiles",
        color_discrete_map={"Success": PLUM, "Failure": SAGE_DARK},
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=520, legend_title_text="")
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    var_ratio = pca.explained_variance_ratio_
    c1, c2, c3 = st.columns(3)
    c1.metric("Variance captured by PC1", f"{var_ratio[0]:.2%}")
    c2.metric("Variance captured by PC2", f"{var_ratio[1]:.2%}")
    c3.metric("Total variance in 2D",
              f"{var_ratio.sum():.2%}",
              help="What's left (~95%) lives in higher dimensions.")

    st.info(
        """
        **How to read this:**

        The two clouds (plum = Success, sage = Failure) are **completely
        overlapping**. There's no region of the plot where you could draw a
        boundary and reliably separate the classes — and that's exactly what
        every classifier we trained discovered the hard way.

        A caveat: PCA only captures the directions of *highest variance*, not
        the directions of *highest predictive value*. So in principle, signal
        could hide in the discarded dimensions. But our mutual information
        analysis already ruled that out: across all 72 dimensions, no feature
        carries meaningful information about the target.
        """
    )


# =========================================================================
# TAB 3 — THE VERDICT
# =========================================================================
with tab3:
    st.markdown("### Putting it all together")

    st.markdown(
        f"""
        We started with a question: **can we predict whether a profile leads
        to a successful match on a dating app, using behavioural and
        demographic features?**

        Three independent lines of evidence say no.
        """
    )

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(
            f"""
            <div style='background:{CREAM}; padding:1.2rem; border-radius:12px;
                        border-left:5px solid {PLUM}; height:230px;'>
                <div style='font-size:1.8rem;'>1️⃣</div>
                <div style='font-weight:bold; color:{PLUM}; font-size:1.1rem;
                            margin:0.4rem 0;'>Five models, same result</div>
                <div style='color:{INK}; font-size:0.92rem;'>
                    Linear, distance-based, tree, ensemble, and kernel methods
                    all converged to within ~1 percentage point of 50%. When
                    fundamentally different algorithms agree, the bottleneck
                    is the data, not the model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            f"""
            <div style='background:{CREAM}; padding:1.2rem; border-radius:12px;
                        border-left:5px solid {SAGE_DARK}; height:230px;'>
                <div style='font-size:1.8rem;'>2️⃣</div>
                <div style='font-weight:bold; color:{SAGE_DARK}; font-size:1.1rem;
                            margin:0.4rem 0;'>Mutual information is near zero</div>
                <div style='color:{INK}; font-size:0.92rem;'>
                    Across the {len(organic_mi)} demographic and behavioural
                    features, the most informative one carries only
                    {organic_mi['mi_score'].max():.4f} bits about the target.
                    A useful predictor would carry 0.1&ndash;0.5 bits. The
                    signal isn&rsquo;t hidden &mdash; it&rsquo;s absent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            f"""
            <div style='background:{CREAM}; padding:1.2rem; border-radius:12px;
                        border-left:5px solid {ACCENT_ROSE}; height:230px;'>
                <div style='font-size:1.8rem;'>3️⃣</div>
                <div style='font-weight:bold; color:{ACCENT_ROSE}; font-size:1.1rem;
                            margin:0.4rem 0;'>Classes overlap in 2D</div>
                <div style='color:{INK}; font-size:0.92rem;'>
                    PCA projection shows Success and Failure profiles
                    occupying the same region of feature space. There is no
                    decision boundary that meaningfully separates them.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### Why this is actually a useful finding")
    st.markdown(
        """
        It's tempting to think of "we couldn't predict it" as a failure. It
        isn't. In real machine learning work, **knowing the ceiling of what's
        possible from a given dataset is one of the most valuable things you
        can deliver**. Three reasons:

        1. **It saves wasted engineering.** A team that doesn't run this
           analysis might spend months tuning models, when no amount of tuning
           can recover information that isn't there.
        2. **It tells you what data you'd actually need.** Match outcomes are
           clearly driven by something this dataset doesn't capture — likely
           conversational quality, mutual chemistry, photo aesthetics, timing,
           and other features that simple swipe behaviour can't proxy.
        3. **It calibrates expectations.** Any product team building on top of
           a "match prediction" model based on this kind of data should know
           upfront that the model is essentially decorative.
        """
    )

    st.markdown("### What would it take to do better?")
    st.markdown(
        """
        Hypothetically, a dataset that could move the needle would need to
        include features like:

        - **Conversation content** — sentiment, response latency,
          message-length asymmetry, topic embedding distance
        - **Photo features** — image embeddings rather than just photo count
        - **Bilateral compatibility signals** — both users' shared
          preferences and overlap, not just one user's profile
        - **Temporal patterns** — how interaction quality evolved over time

        None of those exist in our dataset. The synthetic generator that
        produced this data made every behavioural feature roughly independent
        of the target, which is — ironically — exactly the right setup to
        demonstrate the limits of feature-based prediction.
        """
    )

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{SAGE_DARK}; "
        f"font-style:italic; padding:1rem 0;'>"
        f"Sometimes the most rigorous answer in machine learning is &lsquo;the data "
        f"doesn&rsquo;t support the question.&rsquo;</div>",
        unsafe_allow_html=True,
    )
"""
EDA Dashboard — exploratory data analysis with interactive charts.
Three tabs: Demographics, Behaviour, Outcomes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_eda_sample, load_metadata, load_encoders,
    apply_plotly_theme, inject_global_css,
    PLUM, SAGE_DARK, ACCENT_ROSE, INK,
)

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
inject_global_css()

# --- Header ---
st.markdown(f"<h1 style='color:{PLUM};'>📊 EDA Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "*Exploring 50,000 user profiles across demographics, behaviour, and outcomes.*"
)
st.markdown("---")

# --- Load data ---
df = load_eda_sample().copy()
meta = load_metadata()
encoders = load_encoders()

# Decode label-encoded categorical columns back to human-readable strings.
# This is purely for display on this page — encoded values still exist in
# the artifacts and are used elsewhere (e.g., the Predictor page).
for col, le in encoders.items():
    if col in df.columns:
        df[col] = le.inverse_transform(df[col].astype(int))

# --- Top-level filter (applies to all tabs) ---
with st.expander("🔍 Filter the data", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect(
            "Gender", options=sorted(df["gender"].unique()),
            default=sorted(df["gender"].unique()),
        )
    with col2:
        age_range = st.slider(
            "Age range",
            int(df["age"].min()), int(df["age"].max()),
            (int(df["age"].min()), int(df["age"].max())),
        )

# Note: gender was label-encoded in train.py, so values are integers here.
# We just filter on those integers — labels are recovered via the encoder
# only on the Predictor page.
df_filtered = df[
    (df["gender"].isin(gender_filter))
    & (df["age"].between(age_range[0], age_range[1]))
]

st.caption(f"Showing **{len(df_filtered):,}** profiles "
           f"(out of {len(df):,} sampled from the full 50,000).")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["👥 Demographics", "📱 Behaviour", "💞 Outcomes"])


# =========================================================================
# TAB 1 — DEMOGRAPHICS
# =========================================================================
with tab1:
    st.markdown("### Who is on the app?")

    c1, c2 = st.columns(2)

    with c1:
        # Age distribution
        fig = px.histogram(
            df_filtered, x="age", nbins=30,
            title="Age Distribution",
            color_discrete_sequence=[PLUM],
        )
        fig.update_layout(bargap=0.1, yaxis_title="Number of users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    with c2:
        # Height distribution
        fig = px.histogram(
            df_filtered, x="height_cm", nbins=30,
            title="Height Distribution (cm)",
            color_discrete_sequence=[SAGE_DARK],
        )
        fig.update_layout(bargap=0.1, yaxis_title="Number of users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Zodiac sign breakdown
        zodiac_counts = df_filtered["zodiac_sign"].value_counts().reset_index()
        zodiac_counts.columns = ["zodiac", "count"]
        fig = px.bar(
            zodiac_counts, x="zodiac", y="count",
            title="Zodiac Sign Distribution",
            color_discrete_sequence=[ACCENT_ROSE],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    with c4:
        # Body type
        body_counts = df_filtered["body_type"].value_counts().reset_index()
        body_counts.columns = ["body_type", "count"]
        fig = px.bar(
            body_counts, x="body_type", y="count",
            title="Body Type Distribution",
            color_discrete_sequence=[PLUM],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.info(
        "**Why it matters:** A balanced demographic spread is a precondition for "
        "meaningful modelling. If 90% of users were one gender or age, our model "
        "would learn that bias instead of real behavioural signal."
    )


# =========================================================================
# TAB 2 — BEHAVIOUR
# =========================================================================
with tab2:
    st.markdown("### How do users behave on the app?")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df_filtered, x="app_usage_time_min", nbins=40,
            title="App Usage Time per Day (minutes)",
            color_discrete_sequence=[PLUM],
        )
        fig.update_layout(bargap=0.05, yaxis_title="Users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    with c2:
        fig = px.histogram(
            df_filtered, x="swipe_right_ratio", nbins=30,
            title="Swipe-Right Ratio",
            color_discrete_sequence=[SAGE_DARK],
        )
        fig.update_layout(bargap=0.05, yaxis_title="Users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.scatter(
            df_filtered.sample(min(2000, len(df_filtered))),
            x="likes_received", y="mutual_matches",
            opacity=0.4,
            title="Likes Received vs Mutual Matches",
            color_discrete_sequence=[PLUM],
        )
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    with c4:
        fig = px.histogram(
            df_filtered, x="message_sent_count", nbins=40,
            title="Messages Sent",
            color_discrete_sequence=[ACCENT_ROSE],
        )
        fig.update_layout(bargap=0.05, yaxis_title="Users")
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    # Activity by hour
    hour_counts = df_filtered["last_active_hour"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["hour", "count"]
    fig = px.line(
        hour_counts, x="hour", y="count",
        title="Last Active Hour (24h clock)",
        markers=True,
        color_discrete_sequence=[PLUM],
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Users")
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.info(
        "**Why it matters:** Behavioural features were our biggest hope for "
        "predicting match outcomes. Spoiler: their signal is weak. Mutual "
        "Information analysis on the Insights page shows none of these "
        "exceed 0.01 bits."
    )


# =========================================================================
# TAB 3 — OUTCOMES
# =========================================================================
with tab3:
    st.markdown("### The target variable: Match Outcomes")

    # Class distribution from full dataset (saved in metadata)
    class_dist = pd.DataFrame(
        list(meta["class_distribution"].items()),
        columns=["outcome", "count"],
    ).sort_values("count", ascending=True)

    success_outcomes = meta["success_outcomes"]
    class_dist["category"] = class_dist["outcome"].apply(
        lambda x: "Success" if x in success_outcomes else "Failure"
    )

    fig = px.bar(
        class_dist, x="count", y="outcome", color="category", orientation="h",
        title="All 10 Match Outcomes (full 50,000-row dataset)",
        color_discrete_map={"Success": PLUM, "Failure": SAGE_DARK},
    )
    fig.update_layout(yaxis_title="", xaxis_title="Users",
                      legend_title_text="Category")
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.markdown(
        f"""
        We collapsed these 10 outcomes into a **binary target**:

        - **Success (1)** = {", ".join(success_outcomes)}
        - **Failure (0)** = everything else
        """
    )

    # Binary class balance
    binary_dist = class_dist.groupby("category")["count"].sum().reset_index()
    c1, c2 = st.columns([1, 1.3])
    with c1:
        fig = px.pie(
            binary_dist, values="count", names="category", hole=0.5,
            title="Binary Class Split (raw)",
            color="category",
            color_discrete_map={"Success": PLUM, "Failure": SAGE_DARK},
        )
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
    with c2:
        st.markdown("#### Note on class balancing")
        st.markdown(
            """
            The raw split is roughly **40% Success / 60% Failure**. To prevent
            the models from cheating by always guessing "Failure", we **balanced
            the dataset 50/50** by undersampling the majority class before training.

            This is why "chance level" for our trained models is exactly **50%**,
            not 60%. Any model performing meaningfully above 50% would be finding
            real signal. None of ours do.
            """
        )

    st.info(
        "**Why it matters:** Understanding what we're predicting — and what "
        "'chance' means after balancing — is essential to interpreting the "
        "Model Comparison page correctly."
    )
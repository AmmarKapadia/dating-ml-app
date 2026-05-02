"""
app.py — Entry point + Home page.

Run with:
    streamlit run app.py
"""

import streamlit as st
from utils import load_metadata, inject_global_css, PLUM, SAGE_DARK, INK

# --- Page config (must be the first Streamlit call) ---
st.set_page_config(
    page_title="Tying the Data Knot",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

# --- Load metadata for the stats strip ---
meta = load_metadata()

# --- Header ---
st.markdown(
    f"""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='color: {PLUM}; font-family: Georgia, serif;
                   font-size: 3.2rem; margin-bottom: 0.2rem;'>
            Tying the Data Knot
        </h1>
        <p style='color: {SAGE_DARK}; font-size: 1.3rem; font-style: italic;
                  margin-top: 0;'>
            Love, Life &amp; Likes — A Machine Learning Investigation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --- The pitch ---
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown("### The Question")
    st.markdown(
        """
        On a dating app with **50,000 users** and **19 behavioural features** —
        from swipe ratios to bio length, from emoji usage to last active hour —
        can a machine learning model predict whether two profiles will lead to
        a *successful match*?

        We trained five different classifiers, tuned their hyperparameters,
        and ran proper cross-validation. They all converged to roughly the
        same accuracy.

        **About 50%. Chance.**

        This app tells the story of *why*.
        """
    )

with col_right:
    st.markdown("### At a Glance")
    st.metric("Records analysed", f"{meta['n_train'] + meta['n_test']:,}")
    st.metric("Features engineered", meta["metrics"][meta["best_model"]] and len(meta["feature_names"]))
    st.metric("Models compared", len(meta["metrics"]))
    st.metric("Best model accuracy", f"{meta['metrics'][meta['best_model']]['accuracy']:.1%}")

st.markdown("---")

# --- Navigation guide ---
st.markdown("### Explore the Investigation")
nav_cols = st.columns(4)
nav_items = [
    ("📊", "EDA Dashboard", "How users behave on the app — interactive charts."),
    ("🔮", "Predictor", "Build a profile and see what the model would say."),
    ("⚖️", "Model Comparison", "All five models, side by side, with metrics."),
    ("🔬", "Insights", "The smoking gun: why every model converges to chance."),
]
for col, (emoji, title, desc) in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f"""
            <div style='background:#E8EDE4; padding:1.2rem; border-radius:12px;
                        height:160px;'>
                <div style='font-size:2rem;'>{emoji}</div>
                <div style='font-weight:bold; color:{INK}; margin:0.4rem 0;'>{title}</div>
                <div style='color:{INK}; font-size:0.9rem; opacity:0.8;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("Use the sidebar on the left to navigate between pages.")

st.markdown("---")

# --- Group members ---
st.markdown("### The Team")
st.markdown("**WIA1006 / WID3006 — Machine Learning, Sem 2 2025/2026**")
st.markdown("**Faculty of Computer Science & Information Technology, Universiti Malaya**")

team_cols = st.columns(3)
members = [
    ("Ammar Kapadia", "Group Leader"),
    ("Tan Ker Li", "Member"),
    ("Yim Zi Hao", "Member"),
    ("Chew Chen Xi", "Member"),
    ("Ng Tze Fhung", "Member"),
    ("Muhaimin Afif Bin Mushahar", "Member"),
]
for i, (name, role) in enumerate(members):
    with team_cols[i % 3]:
        st.markdown(
            f"""
            <div style='padding:0.8rem 0;'>
                <div style='font-weight:bold; color:{PLUM};'>{name}</div>
                <div style='color:{SAGE_DARK}; font-size:0.85rem;
                            font-style:italic;'>{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("⚠️ Honest disclaimer: This model achieves only ~50% accuracy "
           "(chance level on a balanced binary task). This is a feature of "
           "this app — we demonstrate *why* with mutual information analysis "
           "on the Insights page.")
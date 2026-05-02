"""
utils.py — Shared loaders and constants for all pages.
Cached with @st.cache_resource so artifacts load once per session.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")

# --- Brand palette (use in plotly charts to match the Streamlit theme) ---
PLUM = "#6B4E71"
SAGE = "#A8B89A"
SAGE_DARK = "#7A8A6E"
WARM_WHITE = "#FAF7F2"
CREAM = "#E8EDE4"
INK = "#2E2A2E"
ACCENT_ROSE = "#C9A0A6"

# Plotly colorway used across all charts for visual consistency
COLORWAY = [PLUM, SAGE_DARK, ACCENT_ROSE, "#8B7C9E", "#B8A89A", "#6E7E62"]


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return joblib.load(ARTIFACTS_DIR / "models.pkl")


@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    return joblib.load(ARTIFACTS_DIR / "scaler.pkl")


@st.cache_resource(show_spinner="Loading encoders...")
def load_encoders():
    return joblib.load(ARTIFACTS_DIR / "encoders.pkl")


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata():
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading PCA...")
def load_pca():
    pca = joblib.load(ARTIFACTS_DIR / "pca.pkl")
    X_pca = np.load(ARTIFACTS_DIR / "X_train_pca.npy")
    y = np.load(ARTIFACTS_DIR / "y_train.npy")
    return pca, X_pca, y


@st.cache_data(show_spinner="Loading EDA sample...")
def load_eda_sample():
    return pd.read_parquet(ARTIFACTS_DIR / "eda_sample.parquet")


@st.cache_data(show_spinner="Loading mutual information...")
def load_mutual_information():
    return pd.read_csv(ARTIFACTS_DIR / "mutual_information.csv")


def apply_plotly_theme(fig):
    """Style any plotly figure to match the app theme."""
    fig.update_layout(
        colorway=COLORWAY,
        paper_bgcolor=WARM_WHITE,
        plot_bgcolor=WARM_WHITE,
        font=dict(family="Georgia, serif", color=INK, size=13),
        title=dict(font=dict(size=18, color=INK)),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=CREAM, zerolinecolor=CREAM)
    fig.update_yaxes(gridcolor=CREAM, zerolinecolor=CREAM)
    return fig

def inject_global_css():
    """
    Inject app-wide CSS overrides. Call this once near the top of every page
    (after st.set_page_config). Hides the Streamlit decoration bar, polishes
    a few small visual details, and aligns spacing with the theme.
    """
    import streamlit as st
    st.markdown(
        f"""
        <style>
            /* Hide the rainbow gradient decoration bar at the top of every page */
            div[data-testid="stDecoration"] {{ display: none; }}

            /* Hide the "Made with Streamlit" footer (cleaner for the demo video) */
            footer {{ visibility: hidden; }}

            /* Hide the hamburger menu (we control the experience, not the user) */
            #MainMenu {{ visibility: hidden; }}

            /* Tighter top padding on every page so titles sit closer to the top */
            .block-container {{ padding-top: 2rem; }}

	    /* Slider track stays plum */
            .stSlider [data-baseweb="slider"] > div > div {{
                background-color: {PLUM} !important;
            }}

	    /* Min/max range labels: clean white pill with dark text.
               High-specificity selectors to beat Streamlit's internal
               st-emotion-cache classes in the cascade. */
            div[data-testid="stSliderTickBar"] div[data-testid="stSliderTickBarMin"],
            div[data-testid="stSliderTickBar"] div[data-testid="stSliderTickBarMax"] {{
                background: #FFFFFF !important;
                background-color: #FFFFFF !important;
                color: {INK} !important;
                opacity: 1 !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                padding: 2px 10px !important;
                border-radius: 4px !important;
                border: 1px solid #E0DDD8 !important;
                box-shadow: none !important;
            }}

            /* The little number bubble above the slider handle — keep it
               readable but neutral, not plum-on-plum */
            .stSlider [data-baseweb="slider"] [role="slider"] + div {{
                color: {INK} !important;
            }}

            /* Make st.metric labels a touch more elegant */
            [data-testid="stMetricLabel"] {{
                color: {SAGE_DARK} !important;
                font-style: italic;
                font-size: 0.95rem;
            }}
            [data-testid="stMetricValue"] {{
                color: {INK} !important;
                font-family: Georgia, serif;
            }}

            /* Tabs: bigger, theme-coloured underline on the active tab */
            .stTabs [data-baseweb="tab"] {{
                font-size: 1.05rem;
                padding: 0.6rem 1.2rem;
            }}
            .stTabs [aria-selected="true"] {{
                color: {PLUM} !important;
            }}
            .stTabs [data-baseweb="tab-highlight"] {{
                background-color: {PLUM} !important;
                height: 3px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
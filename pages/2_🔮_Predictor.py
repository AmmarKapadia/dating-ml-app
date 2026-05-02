"""
Predictor — interactive form that builds a profile and runs a live prediction.

Loads the saved encoders, scaler, and best model from artifacts/ and uses
them to transform user input the same way the training data was transformed.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import (
    load_models, load_scaler, load_encoders, load_metadata,
    apply_plotly_theme, inject_global_css,
    PLUM, SAGE_DARK, ACCENT_ROSE, CREAM, INK,
)

st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
inject_global_css()

st.markdown(f"<h1 style='color:{PLUM};'>🔮 The Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("*Build a profile and see what the model would predict — "
            "then read the honest disclaimer below.*")
st.markdown("---")

# --- Load artifacts ---
models = load_models()
scaler = load_scaler()
encoders = load_encoders()
meta = load_metadata()

best_model_name = meta["best_model"]
feature_names = meta["feature_names"]
interest_columns = meta["interest_columns"]

# --- Model selector ---
model_choice = st.selectbox(
    "Which model should make the prediction?",
    options=list(models.keys()),
    index=list(models.keys()).index(best_model_name),
    help="Defaults to the best-performing model from training.",
)
model = models[model_choice]

st.markdown("---")

# =========================================================================
# THE FORM — three columns to keep it compact
# =========================================================================
st.markdown("### Build your profile")

# Helper: get original labels for an encoded column
def labels_for(col):
    return list(encoders[col].classes_)

c1, c2, c3 = st.columns(3)

# --- Column 1: Demographics ---
with c1:
    st.markdown("**👤 Demographics**")
    gender = st.selectbox("Gender", labels_for("gender"))
    sexual_orientation = st.selectbox("Sexual orientation",
                                      labels_for("sexual_orientation"))
    location_type = st.selectbox("Location type", labels_for("location_type"))
    age = st.slider("Age", 18, 70, 28)
    height_cm = st.slider("Height (cm)", 140, 210, 170)
    weight_kg = st.slider("Weight (kg)", 40, 130, 65)
    body_type = st.selectbox("Body type", labels_for("body_type"))
    zodiac_sign = st.selectbox("Zodiac sign", labels_for("zodiac_sign"))

# --- Column 2: Background (drives engineered features) ---
with c2:
    st.markdown("**🎓 Background**")
    income_bracket = st.selectbox(
        "Income bracket",
        ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High", "Very High"],
        index=2,
    )
    education_level = st.selectbox(
        "Education level",
        ["No Formal Education", "High School", "Diploma",
         "Bachelor's", "Master's", "PhD", "Postdoc"],
        index=3,
    )
    relationship_intent = st.selectbox(
        "Relationship intent",
        ["Hookups", "Friends Only", "Casual Dating",
         "Serious Relationship", "Open to Anything"],
        index=2,
    )

    st.markdown("**🎯 Interests** *(pick any)*")
    selected_interests = st.multiselect(
        "Interest tags",
        options=interest_columns,
        default=["Movies", "Traveling"] if "Movies" in interest_columns else [],
        label_visibility="collapsed",
    )

# --- Column 3: Behaviour ---
with c3:
    st.markdown("**📱 App behaviour**")
    app_usage_time_min = st.slider("App usage per day (min)", 0, 600, 90)
    app_usage_time_label = st.selectbox(
        "Usage label", labels_for("app_usage_time_label"),
    )
    swipe_right_ratio = st.slider("Swipe-right ratio", 0.0, 1.0, 0.5, 0.01)
    swipe_right_label = st.selectbox(
        "Swipe label", labels_for("swipe_right_label"),
    )
    swipe_time_of_day = st.selectbox(
        "Active swipe time", labels_for("swipe_time_of_day"),
    )
    likes_received = st.slider("Likes received", 0, 500, 60)
    mutual_matches = st.slider("Mutual matches", 0, 100, 8)
    profile_pics_count = st.slider("Profile pictures", 0, 10, 4)
    bio_length = st.slider("Bio length (chars)", 0, 500, 120)
    message_sent_count = st.slider("Messages sent", 0, 500, 40)
    emoji_usage_rate = st.slider("Emoji usage rate", 0.0, 1.0, 0.3, 0.01)
    last_active_hour = st.slider("Last active hour (0-23)", 0, 23, 20)


# =========================================================================
# BUILD THE FEATURE ROW
# =========================================================================
def build_input_row():
    """Construct a single-row DataFrame matching the training feature layout."""
    HIGH_INCOME = ["High", "Very High", "Upper-Middle", "Middle"]
    HIGH_EDU = ["Bachelor's", "Master's", "PhD", "Postdoc"]
    HIGH_INTENT = ["Serious Relationship", "Casual Dating"]

    # Start with a row of zeros for every feature the model expects
    row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Fill numeric columns directly
    numeric_inputs = {
        "age": age, "height_cm": height_cm, "weight_kg": weight_kg,
        "app_usage_time_min": app_usage_time_min,
        "swipe_right_ratio": swipe_right_ratio,
        "likes_received": likes_received,
        "mutual_matches": mutual_matches,
        "profile_pics_count": profile_pics_count,
        "bio_length": bio_length,
        "message_sent_count": message_sent_count,
        "emoji_usage_rate": emoji_usage_rate,
        "last_active_hour": last_active_hour,
    }
    for col, val in numeric_inputs.items():
        if col in row.columns:
            row.at[0, col] = val

    # Encode categorical columns using the saved encoders
    categorical_inputs = {
        "gender": gender,
        "sexual_orientation": sexual_orientation,
        "location_type": location_type,
        "app_usage_time_label": app_usage_time_label,
        "swipe_right_label": swipe_right_label,
        "swipe_time_of_day": swipe_time_of_day,
        "zodiac_sign": zodiac_sign,
        "body_type": body_type,
    }
    for col, val in categorical_inputs.items():
        if col in row.columns:
            row.at[0, col] = encoders[col].transform([val])[0]

    # Engineered binary features
    if "feat_income_high" in row.columns:
        row.at[0, "feat_income_high"] = int(income_bracket in HIGH_INCOME)
    if "feat_edu_high" in row.columns:
        row.at[0, "feat_edu_high"] = int(education_level in HIGH_EDU)
    if "feat_intent_high" in row.columns:
        row.at[0, "feat_intent_high"] = int(relationship_intent in HIGH_INTENT)

    # Interest tag one-hots
    for interest in selected_interests:
        if interest in row.columns:
            row.at[0, interest] = 1

    return row


# =========================================================================
# PREDICT
# =========================================================================
st.markdown("---")
predict_col, _ = st.columns([1, 3])
with predict_col:
    predict_clicked = st.button("🔮 Predict match outcome",
                                type="primary", use_container_width=True)

if predict_clicked:
    input_row = build_input_row()
    input_scaled = scaler.transform(input_row)

    # Get prediction + probability if the model supports it
    pred = int(model.predict(input_scaled)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        prob_failure, prob_success = float(proba[0]), float(proba[1])
    else:
        # SVC with default settings doesn't expose probabilities; fallback
        prob_success = float(pred)
        prob_failure = 1.0 - prob_success

    label = "Successful Match" if pred == 1 else "Unlikely to Match"
    accent = PLUM if pred == 1 else SAGE_DARK
    emoji = "💞" if pred == 1 else "🌫️"

    # --- Result card ---
    st.markdown(
        f"""
        <div style='background:{CREAM}; padding:1.8rem; border-radius:14px;
                    border-left:6px solid {accent}; margin-top:1rem;'>
            <div style='font-size:2.2rem;'>{emoji}</div>
            <div style='font-size:1.6rem; color:{accent}; font-weight:bold;
                        margin-top:0.3rem;'>{label}</div>
            <div style='color:{INK}; opacity:0.7; margin-top:0.3rem;'>
                Predicted by <em>{model_choice}</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Probability gauge ---
    st.markdown("#### How confident is the model?")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_success * 100,
        number={"suffix": "%", "font": {"color": INK, "size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": INK,
                     "tickfont": {"color": INK}},
            "bar": {"color": PLUM, "thickness": 0.25},
            "bgcolor": CREAM,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "#D4DCC8"},
                {"range": [45, 55], "color": "#E8EDE4"},
                {"range": [55, 100], "color": "#D8C5DA"},
            ],
            "threshold": {
                "line": {"color": ACCENT_ROSE, "width": 3},
                "thickness": 0.85, "value": 50,
            },
        },
    ))
    # Theme the gauge manually — apply_plotly_theme conflicts with Indicator's title
    gauge_fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor=CREAM,
        font=dict(family="Georgia, serif", color=INK, size=13),
        title={
            "text": "Probability of a successful match",
            "x": 0.5, "xanchor": "center",
            "font": {"color": INK, "size": 16},
        },
    )
    st.plotly_chart(gauge_fig, use_container_width=True)
    # --- Probability bar ---
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.metric("Probability of Success", f"{prob_success:.1%}")
    with pcol2:
        st.metric("Probability of Failure", f"{prob_failure:.1%}")

    # --- The honest disclaimer ---
    st.warning(
        f"""
        **🔬 Honest interpretation**

        This prediction comes from a model with ~{meta['metrics'][model_choice]['accuracy']:.0%}
        test-set accuracy — essentially **chance** on a balanced binary task.

        Across all five models we trained, none performed meaningfully above
        50%. Mutual information analysis (see the Insights page) shows that
        no feature in this dataset carries more than ~0.01 bits of information
        about the target.

        **In other words: this prediction is theatre, not science.**
        That's the honest finding of our project — the dataset itself doesn't
        contain enough signal to predict match outcomes, regardless of which
        model you throw at it.
        """
    )

else:
    st.info(
        "👆 Fill in the form above and click **Predict match outcome** "
        "to see what the model would say. Then check the Insights page "
        "to learn why you should not, under any circumstances, take this "
        "prediction seriously."
    )
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ✅ Load model and scaler with caching to reduce startup time
@st.cache_resource
def load_model_and_scaler():
   model = joblib.load("Celebal-main-Project/saved_models/spam_model.pkl")
   scaler = joblib.load("Celebal-main-Project/saved_models/scaler.pkl")

   return model, scaler

model, scaler = load_model_and_scaler()

# Set page config
st.set_page_config(page_title="Email Spam Classifier", layout="wide")

# Custom CSS for theming and icons
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .st-emotion-cache-1v0mbdj { padding-top: 0rem; }
        .result-box {
            padding: 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            margin-top: 1rem;
        }
        .not-spam {
            background-color: #dcfce7;
            color: #15803d;
            border-left: 5px solid #22c55e;
        }
        .spam {
            background-color: #fee2e2;
            color: #b91c1c;
            border-left: 5px solid #ef4444;
        }
    </style>
""", unsafe_allow_html=True)

# Toggle light/dark theme
theme = st.radio("🌗 Select Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True)
if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
            .main { background-color: #1e1e1e; color: white; }
        </style>
        """, unsafe_allow_html=True
    )

st.title("📧 Email Spam Classifier")

# Tabs
tabs = st.tabs(["🧪 Quick Demo", "🧮 Full Manual Input", "📄 Upload CSV"])

# Sample Inputs
spam_samples = {
    "Spam Sample A": [0.2, 0.4, 0.6, 0.8, 0.1, 0.33, 0.57, 0.72, 0.09, 0.99],
    "Spam Sample B": [0.9, 0.8, 0.85, 0.95, 0.7, 0.75, 0.8, 0.9, 0.88, 0.92]
}
nonspam_samples = {
    "Non-Spam Sample A": [0.1, 0.2, 0.1, 0.2, 0.15, 0.05, 0.07, 0.1, 0.05, 0.08],
    "Non-Spam Sample B": [0.01, 0.03, 0.02, 0.04, 0.01, 0.00, 0.02, 0.01, 0.03, 0.00]
}

# 🧪 Quick Demo Tab
with tabs[0]:
    st.subheader("🎯 Quick 10-Feature Test")

    demo_inputs = [0.0] * 57
    demo_type = st.radio("Choose a Sample:", ["None"] + list(spam_samples) + list(nonspam_samples), horizontal=True)

    if demo_type in spam_samples:
        demo_inputs[:10] = spam_samples[demo_type]
    elif demo_type in nonspam_samples:
        demo_inputs[:10] = nonspam_samples[demo_type]

    user_input = []
    for i in range(10):
        val = st.number_input(f"Feature {i+1}", value=demo_inputs[i], min_value=0.0, max_value=100.0)
        user_input.append(val)

    final_input = user_input + [0.0] * 47

    if st.button("🔍 Classify (Quick Demo)"):
        scaled_input = scaler.transform([final_input])
        result = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        st.markdown(f"**📊 Confidence:** `{prob * 100:.2f}%`")

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 Warning! This email is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✅ This email is NOT Spam.</div>', unsafe_allow_html=True)

# 🧮 Full Manual Input Tab
with tabs[1]:
    st.subheader("🧠 Enter All 57 Features Manually")

    full_inputs = []
    for i in range(57):
        val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=100.0, key=f"full_{i}")
        full_inputs.append(val)

    if st.button("🔍 Classify (Full Input)"):
        scaled = scaler.transform([full_inputs])
        result = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]
        st.markdown(f"**📊 Confidence:** `{prob * 100:.2f}%`")

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 Warning! This email is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✅ This email is NOT Spam.</div>', unsafe_allow_html=True)

# 📄 Upload CSV Tab
with tabs[2]:
    st.subheader("📁 Upload CSV File with 57 Features")

    file = st.file_uploader("Upload a CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if df.shape[1] != 57:
            st.error("❌ CSV must contain exactly 57 columns.")
        else:
            st.info("✅ File uploaded. Preview:")
            st.dataframe(df.head())

            scaled_df = scaler.transform(df)
            preds = model.predict(scaled_df)
            probs = model.predict_proba(scaled_df)[:, 1]

            df["Spam Probability (%)"] = (probs * 100).round(2)
            df["Prediction"] = np.where(preds == 1, "SPAM", "NOT SPAM")

            st.success("🎉 Prediction Completed!")
            st.dataframe(df)

            st.download_button("⬇️ Download Results", df.to_csv(index=False), "spam_predictions.csv", mime="text/csv")

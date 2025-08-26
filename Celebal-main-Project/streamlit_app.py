import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Set page config
st.set_page_config(page_title="Email Spam Classifier", layout="wide")

# ✅ Load model and scaler with caching to reduce startup time
@st.cache_resource
def load_model_and_scaler():
   model = joblib.load("saved_models/spam_model.pkl")
   scaler = joblib.load("saved_models/scaler.pkl")

   return model, scaler

model, scaler = load_model_and_scaler()



# Global Custom CSS
st.markdown("""
    <style>
        /* ============ GLOBAL APP STYLING ============ */
        .stApp {
            font-family: 'Segoe UI', Tahoma, sans-serif !important;
            transition: all 0.3s ease-in-out !important;
        }

        /* Remove padding around Streamlit containers */
        .st-emotion-cache-1v0mbdj {
            padding-top: 0rem !important;
        }

        /* RESULT BOXES */
        .result-box {
            padding: 1rem !important;
            border-radius: 0.8rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        .not-spam {
            background: linear-gradient(135deg, #bbf7d0, #22c55e) !important;
            color: #064e3b !important;
            border-left: 6px solid #065f46 !important;
        }
        .spam {
            background: linear-gradient(135deg, #fecaca, #ef4444) !important;
            color: #7f1d1d !important;
            border-left: 6px solid #991b1b !important;
        }

        /* HEADERS */
        .demo-header {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
            padding: 0.8rem !important;
            border-radius: 0.6rem !important;
            text-align: center !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15) !important;
        }

        .custom-subheader {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            margin-top: 1.2rem !important;
            padding: 0.4rem 0.8rem !important;
            border-radius: 0.3rem !important;
            display: inline-block !important;
            box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important;
        }
            
                /* ============ HOVER & CLICK EFFECTS ============ */
        .stButton>button, .stNumberInput input, .stTextInput input, .stFileUploader {
            transition: all 0.25s ease-in-out !important;
        }

        /* Buttons hover/active */
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.03);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25) !important;
        }
        .stButton>button:active {
            transform: translateY(1px) scale(0.97);
            box-shadow: 0 3px 8px rgba(0,0,0,0.2) !important;
        }

        /* Inputs hover focus */
        .stNumberInput input:hover, .stTextInput input:hover, .stFileUploader:hover {
            border-color: #f59e0b !important;   /* amber highlight */
            box-shadow: 0 0 8px rgba(245,158,11,0.6) !important;
        }
        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: #ef4444 !important;   /* red highlight */
            box-shadow: 0 0 10px rgba(239,68,68,0.7) !important;
            outline: none !important;
        }

    </style>
""", unsafe_allow_html=True)

# Toggle light/dark theme
theme = st.radio("🌗 Select Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True)

# Theme-specific CSS overrides
if theme == "🌞 Light Mode":
    st.markdown("""
        <style>
            body, .stApp {
                background: linear-gradient(135deg, #fef3c7, #fcd34d, #fca5a5) !important;
                background-attachment: fixed !important;
                background-size: cover !important;
                color: #111827 !important;
            }
            h1, h2, h3, h4, h5, h6, p, label {
                color: #1f2937 !important;
            }
            /* Inputs */
            .stNumberInput, .stTextInput, .stFileUploader {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 2px solid #e5e7eb !important;
                border-radius: 10px !important;
                padding: 5px !important;
            }
            /* Buttons */
            .stButton>button {
                background: linear-gradient(90deg, #f59e0b, #ef4444) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 8px 20px !important;
                font-weight: bold !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg, #f87171, #fbbf24) !important;
                box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)

else:  # 🌙 Dark Mode
    st.markdown("""
        <style>
            body, .stApp {
                background: linear-gradient(135deg, #0f172a, #1e293b, #334155) !important;
                background-attachment: fixed !important;
                background-size: cover !important;
                color: #f1f5f9 !important;
            }
            h1, h2, h3, h4, h5, h6, p, label {
                color: #f9fafb !important;
            }
            /* Inputs */
            .stNumberInput, .stTextInput, .stFileUploader {
                background-color: #1e293b !important;
                color: #f1f5f9 !important;
                border: 2px solid #475569 !important;
                border-radius: 10px !important;
                padding: 5px !important;
            }
            /* Buttons */
            .stButton>button {
                background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 8px 20px !important;
                font-weight: bold !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
                box-shadow: 0 6px 12px rgba(0,0,0,0.5) !important;
            }
        </style>
    """, unsafe_allow_html=True)


# App Title

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

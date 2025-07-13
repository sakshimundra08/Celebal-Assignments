import streamlit as st
import pandas as pd
import os
import utils.retriever as retriever
import utils.summarizer as summarizer
from utils.corpus_utils import build_corpus
from PIL import Image
import base64

st.set_page_config(page_title="Loan Insights SmartBot", layout="wide")

# ✅ Inject background GIF
st.markdown(
    """
    <style>
    .stApp {
        background: url('static/bg.gif');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🚫 Removed logo loading block

# Inject custom CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("💬 Loan Insights SmartBot")

# Sidebar - Mode selection
mode = st.sidebar.radio("Choose Mode", ["Chat Mode", "Quiz Mode"])

# Sidebar - File upload
uploaded = st.sidebar.file_uploader("📂 Upload your own CSV", type="csv")
data = None
if uploaded:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv("data/Training Dataset.csv")

# Sidebar - Filters
approved_only = st.sidebar.checkbox("✅ Only Approved")
graduate_only = st.sidebar.checkbox("🎓 Only Graduates")

# Build the document corpus
corpus_df = build_corpus(data, approved_only, graduate_only)
index, corpus = retriever.create_faiss_index(corpus_df["text"].tolist())

# Session state setup
if "history" not in st.session_state:
    st.session_state.history = []

if mode == "Chat Mode":
    st.markdown("### 🧠 Ask me anything about the loan data!")
    query = st.text_input("Your question:", key="input")

    if st.button("Ask") and query:
        top_docs = retriever.query_faiss(index, corpus, query)
        answer = summarizer.summarize_answer(query, top_docs)

        st.session_state.history.append((query, answer))

        # Display chat
        for q, a in st.session_state.history:
            st.markdown(f"💬 **You:** {q}")
            st.markdown(f"🤖 **Bot:** {a}")

elif mode == "Quiz Mode":
    st.markdown("<h3 style='color: #111111;'>🎯 Quiz: Can you guess if the applicant got approved?</h3>", unsafe_allow_html=True)

    quiz_row = corpus_df.sample(1).iloc[0]
    info = quiz_row.drop("text").to_dict()
    actual = "Approved ✅" if quiz_row["Loan_Status"] == "Y" else "Not Approved ❌"

    for k, v in info.items():
        if k != "Loan_Status":
            st.markdown(f"<p style='color: #111111;'><b>{k}:</b> {v}</p>", unsafe_allow_html=True)

    guess = st.radio("Your guess:", ["Approved ✅", "Not Approved ❌"])
    if st.button("Check Answer"):
        st.success(f"Correct answer: **{actual}**")
        if guess == actual:
            st.balloons()
        else:
            st.info("Not quite — check the factors again!")


# Download chat history
if st.button("📄 Export Chat History") and st.session_state.history:
    history_txt = "\n\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.history])
    b64 = base64.b64encode(history_txt.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">Download .txt</a>'
    st.markdown(href, unsafe_allow_html=True)

# 💬 Loan Insights SmartBot

An interactive Streamlit app that helps users explore loan approval data using AI-driven Q&A and quiz features.

## 🚀 Features

- 🤖 **Chat Mode**: Ask natural language questions and get intelligent answers from the loan dataset using sentence embeddings.
- 🎯 **Quiz Mode**: Test your intuition by guessing loan approval outcomes.
- 📊 **Custom Filters**: Filter dataset by approval status and education level.
- 📂 **CSV Upload**: Upload your own dataset to explore.
- 📄 **Export Chat**: Download your chat history in `.txt` format.
- 🌌 **Background & Styling**: Enhanced UI with animated background and custom styling.

## 📁 Folder Structure

celebal week 8/
│
├── loan_insight_smartbot.py # Main Streamlit app
├── requirements.txt # Dependencies
├── utils/
│ ├── corpus_utils.py # Corpus builder logic
│ ├── retriever.py # FAISS-based semantic search
│ └── summarizer.py # Summary generation
├── static/
│ ├── bg.gif # Background animation
│ └── styles.css # Custom styling
└── README.md # Project info

perl
Copy
Edit

## 🛠️ Installation

```bash
pip install -r requirements.txt
streamlit run loan_insight_smartbot.py
📦 Model Used
Sentence Transformers: all-MiniLM-L6-v2 via HuggingFace

🔗 Author
Sakshi Mundra


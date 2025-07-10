# 📧 Email Spam Classifier

A machine learning project to classify emails as **Spam** or **Not Spam** based on 57 numerical features extracted from text. Built with **Random Forest Classifier**, this app allows users to test the model through:
- A quick demo with sample inputs,
- Full manual input (all 57 features),
- CSV upload for batch predictions.

---

## 🧠 Features

- 🎯 Quick Demo Mode with spam & non-spam sample inputs.
- ✍️ Full Manual Entry of all 57 features.
- 📁 CSV Upload support for batch classification.
- 📊 Displays prediction confidence percentage.
- 🧰 Model trained on UCI Spambase Dataset using Random Forest.
- 🌐 Built using Python, Streamlit, Scikit-learn, Joblib.

---

## 🗂 Project Structure

Celebal-main-Project/
│
├── streamlit_app.py # Main Streamlit app
├── spam_classifier.py # Script to train and save model
├── requirements.txt # Dependencies
├── saved_models/
│ ├── spam_model.pkl # Trained Random Forest model
│ └── scaler.pkl # StandardScaler object
└── .gitignore # (Optional) to ignore system files


---

## 🔧 How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/sakshimundra08/Celebal-Assignments.git
   cd Celebal-Assignments/Celebal-main-Project

**Install dependencies**

pip install -r requirements.txt


**Run the app**

streamlit run streamlit_app.py

📊 Dataset Used

UCI Spambase Dataset
Source: UCI Machine Learning Repository

Total Samples: 4601

Features: 57 (word frequency, capital letters, etc.)

Labels: 1 = Spam, 0 = Not Spam

✨ Model Details
Algorithm: Random Forest Classifier

Scaler: StandardScaler

Accuracy: > 94% on test split

Trained using: spam_classifier.py

🙋‍♀️ Author
Sakshi Mundra
GitHub: @sakshimundra08


# Social Media Sentiment Classifier – Week 7 Assignment

🚀 **Live App Link:**  
👉 [https://social-media-engagement1.streamlit.app/](https://social-media-engagement1.streamlit.app/)

✅ Hosted on: [Streamlit Cloud](https://streamlit.io/cloud)  

This is the Week 7 assignment for the Celebal Technologies Data Science program.

It predicts the **sentiment score** of a social media post based on metadata such as platform, post type, likes, comments, 
and shares using a **Random Forest Classifier**. The application is built using **Streamlit** and visualized with **Plotly**.



**Overview**

Predicts sentiment (`positive`, `neutral`, `negative`) of social media posts based on:
- Platform (e.g., Facebook, Twitter)
- Post type (image, video, etc.)
- Day of posting
- Engagement stats (likes, comments, shares)


**Built with:**

- Python, pandas, NumPy
- scikit-learn (RandomForestClassifier)
- Streamlit for the UI
- Plotly for visualizations


**How to Run Locally:**

1. Clone this repo:

git clone https://github.com/sakshimundra08/Celebal-Assignments
cd Celebal-Assignments/week_7_assignment

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run mymodel.py


# Files Included

| File Name                      | Description                              |
|-------------------------------|------------------------------------------|
| `mymodel.py`                  | Streamlit app with model training & UI   |
| `requirements.txt`            | Required Python packages                 |
| `social_media_engagement1.csv`| Dataset used for model training          |

**Output Features:**

i) Real-time sentiment prediction

ii) Input via dropdowns and sliders

iii) Probability distribution chart

iv) Dataset sentiment distribution visualization

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and preprocess the data
@st.cache_resource
def load_data():
    df = pd.read_csv("social_media_engagement1.csv")
    
    # Encode categorical columns
    label_encoders = {}
    for col in ['platform', 'post_type', 'post_day', 'sentiment_score']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Train the model
@st.cache_resource
def train_model(df):
    X = df[['platform', 'post_type', 'post_day', 'likes', 'comments', 'shares']]
    y = df['sentiment_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Streamlit app
def main():
    st.title("📱 Social Media Sentiment Classifier")
    st.write("Predict the **sentiment score** of a social media post based on its metadata.")

    # Load data and model
    df, encoders = load_data()
    model = train_model(df)

    # Input section
    st.subheader("📥 Input Post Details")

    platform = st.selectbox("Platform", encoders['platform'].classes_)
    post_type = st.selectbox("Post Type", encoders['post_type'].classes_)
    post_day = st.selectbox("Day of Posting", encoders['post_day'].classes_)
    likes = st.slider("Number of Likes", 0, int(df['likes'].max()), 1000)
    comments = st.slider("Number of Comments", 0, int(df['comments'].max()), 100)
    shares = st.slider("Number of Shares", 0, int(df['shares'].max()), 100)

    if st.button("Predict Sentiment"):
        # Encode user input
        input_data = pd.DataFrame([{
            'platform': encoders['platform'].transform([platform])[0],
            'post_type': encoders['post_type'].transform([post_type])[0],
            'post_day': encoders['post_day'].transform([post_day])[0],
            'likes': likes,
            'comments': comments,
            'shares': shares
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        prediction_label = encoders['sentiment_score'].inverse_transform([prediction])[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display result
        st.success(f"🔮 Predicted Sentiment: **{prediction_label.upper()}**")

        # Probability chart
        proba_df = pd.DataFrame({
            "Sentiment": encoders['sentiment_score'].classes_,
            "Probability": prediction_proba
        })
        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(proba_df.set_index("Sentiment"))

    # Optional: visualize original dataset distribution
    st.subheader("📈 Sentiment Distribution in Dataset")
    dist_df = df.copy()
    dist_df['sentiment_score'] = encoders['sentiment_score'].inverse_transform(df['sentiment_score'])
    fig = px.histogram(dist_df, x="sentiment_score", color="sentiment_score", title="Sentiment Class Distribution")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

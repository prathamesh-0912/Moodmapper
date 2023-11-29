# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 

import os

# Get the absolute path to the model file
model_path = os.path.abspath("emotion_classifier_pipe.pkl")

# Load the model
pipe_lr = joblib.load(open(model_path, "rb"))

# Set custom title and page configuration
st.set_page_config(
    page_title="Moodmapper - Emotion Classifier App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Define the dictionary for emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# Main Application
st.title("Moodmapper - Emotion Classifier App")
menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home-Emotion In Text")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Apply Fxn Here
        results = pipe_lr.predict([raw_text])
        prediction = results[0]
        probability = pipe_lr.predict_proba([raw_text])

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

elif choice == "About":
    st.subheader("About")
    st.text("Emotion in Text Classifier: Analyzing Emotional Content in Textual Data")
    st.text("Introduction: Emotion in text classifiers is a fascinating application of natural")
    st.text("language processing (NLP) and machine learning that seeks to decipher and categorize")
    st.text("the emotional nuances conveyed through written text. In an age where text")
    st.text("communication is prolific – from social media posts and customer reviews to news")
    st.text("articles and chat conversations – understanding the emotional tone underlying")
    st.text("the text becomes a valuable asset for various fields, from marketing and customer")
    st.text("service to psychology and social sciences.")
    st.text("Understanding Emotion Classification: Emotion classification involves training")
    st.text("algorithms to recognize and categorize different emotional states within a piece of")
    st.text("text. Unlike sentiment analysis, which categorizes text into broad sentiments like")
    st.text("positive, negative, or neutral, emotion classification delves deeper, attempting to")
    st.text("pinpoint specific emotions such as joy, anger, fear, sadness, surprise, and more.")
    st.text("This granularity provides a richer understanding of the emotional context within")
    st.text("text.")

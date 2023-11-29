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
pipe_lr = joblib.load(open("C:\\Users\\lawan\\OneDrive\\Documents\\Projects\\Emotion-in-Text-classifier\\notebbok\\emotion_classifier_pipe.pkl", "rb"))

st.set_page_config(
    page_title="Your App Name",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Emotion Classifier App")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	
	if choice == "Home":
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "About":
		st.subheader("About")
		st.text("Emotion in Text Classifier: Analyzing Emotional Content in Textual Data \nIntroduction: Emotion in text classifiers is a fascinating application of natural \nlanguage processing (NLP) and machine learning that seeks to decipher and categorize \nthe emotional nuances conveyed through written text. In an age where text \ncommunication is prolific – from social media posts and customer reviews to news \narticles and chat conversations – understanding the emotional tone underlying \nthe text becomes a valuable asset for various fields, from marketing and customer \nservice to psychology and social sciences. \n \nUnderstanding Emotion Classification: Emotion classification involves training \nalgorithms to recognize and categorize different emotional states within a piece of \ntext. Unlike sentiment analysis, which categorizes text into broad sentiments like \npositive, negative, or neutral, emotion classification delves deeper, attempting to \npinpoint specific emotions such as joy, anger, fear, sadness, surprise, and more. \nThis granularity provides a richer understanding of the emotional context within \ntext.")

if __name__ == '__main__':
	main()
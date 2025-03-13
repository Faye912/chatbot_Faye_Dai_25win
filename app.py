import streamlit as st
import requests

st.title("Emotion Detection")

user_input = st.text_area("Enter a sentence:")

if st.button("Analyze Emotion"):
    response = requests.post("http://127.0.0.1:8000/predict/", json={"text": user_input})
    
    if response.status_code == 200:
        st.write(f"Predicted Emotion: {response.json()['predicted_emotion']}")
    else:
        st.write("Error in prediction.")





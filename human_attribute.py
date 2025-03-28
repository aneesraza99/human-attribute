mkdir human-attribute
   cd human-attribute

import streamlit as st
import google.generativeai as genai
import os
import PIL.Image

# Configure Gemini (API key will be set via Streamlit secrets)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load the Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def analyze_human_attributes(image):
    prompt = """
    Analyze the image and return these exact details:
    - Gender (Male/Female/Non-binary)
    - Age Estimate (e.g., 25 years)
    - Ethnicity (e.g., Asian, Caucasian, African)
    - Mood (e.g., Happy, Sad, Neutral)
    - Facial Expression (e.g., Smiling, Frowning)
    - Glasses (Yes/No)
    - Beard (Yes/No)
    - Hair Color (e.g., Black, Blonde)
    - Eye Color (e.g., Blue, Brown)
    - Headwear (Yes/No)
    - Emotions Detected (e.g., Joyful, Angry)
    - Confidence Level (Percentage)
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()

# Streamlit App
st.title("Human Attribute Detection")
uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    person_info = analyze_human_attributes(img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_column_width=True)
    with col2:
        st.write(person_info)
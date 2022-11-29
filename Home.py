import streamlit as st
from PIL import Image
image = Image.open( "house.jpeg")

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Welcome to Property Damage Prediction For Weather Disasters")

st.image(image, caption='Each year 25.34 Billion dollars of propery damages occur from weather disasters')
st.subheader("Our Mission:")
st.text('To provide reliable prediction of propery damages caused by weather disasters to protect and serve those potentially at risk  ')





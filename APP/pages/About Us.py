import streamlit as st
from PIL import Image


st.markdown("<h1 style='text-align: center; color: black;'>Meet Our Team</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Innovation For The Better</h2>", unsafe_allow_html=True)




image = Image.open( "111.png")
st.image(image, width= 750)




image = Image.open( "z.jpg")
st.image(image, width= 700)



st.subheader("Dedicated to delivering accurate property damage predictions to help our users")





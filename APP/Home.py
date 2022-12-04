import streamlit as st
from PIL import Image

image = Image.open( "house.jpeg")

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋"
)


st.title("Welcome to Property Damage Prediction For Weather Disasters")

st.image(image)
original_title = '<p style="font-family:Sanif; color:Black; font-size: 25spx;">Each year 25.34 Billion dollars of propery damages occur from weather disasters</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.title("Summary")
original_title = '<p style="font-family:Sanif; color:Black; font-size: 19px;">Tropical storms affect multiple different aspects of life. It impacts the environment, infrastructure, agriculture, economy, and most important of all human life. There have been 56 different weather and climate disasters in the US in the past three years costing more than $2 trillion in damages out of which most of the damages were done by tropical storms. Being able to predict these tropical storms can help a lot. Every year a large number of people are affected by these storms. Many people permanently lose their homes and all the belongings that they worked hard to build and acquire, in some serious cases people lose their lives and their loved ones.  Predictive models can help us determine when, where, and how hard a tropical storm is going to occur and precautions can be taken accordingly. If we have accurate predictions we can use that to make sure people are in safe and secured shelters which will reduce the loss of life, proper planning can take place by the government to keep any aid and resources available in nearby areas to help people.</p>'
st.markdown(original_title, unsafe_allow_html=True)



st.title("Get Started")
original_title = '<p style="font-family:Sanif; color:Black; font-size: 19px;">Dashboard Page: We provide an interactive dasboard with property damage information based on the weather event, month, and area </p>'
st.markdown(original_title, unsafe_allow_html=True)


original_title = '<p style="font-family:Sanif; color:Black; font-size: 19px;">Prediction Page: Select a County and State, weather event, month, and year to recieve an estimated cost prediction of property damages </p>'
st.markdown(original_title, unsafe_allow_html=True)

original_title = '<p style="font-family:Sanif; color:Black; font-size: 19px;">Data Source: Storm Events Data From NOAA</p>'
st.markdown(original_title, unsafe_allow_html=True)

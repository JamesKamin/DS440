import streamlit as st


st.title("Contact")

with st.form("form1", clear_on_submit=True):
    name = st.text_input("Enter full name")
    st.form_submit_button()


with st.form("form2", clear_on_submit=True):
    email = st.text_input("Enter Email")
    st.form_submit_button()


with st.form("form3", clear_on_submit=True):
    message = st.text_area("Message")
    st.form_submit_button()



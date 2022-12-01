
import streamlit as st

from sklearn.linear_model import Lasso

from sklearn.preprocessing import LabelEncoder

import joblib 

import pandas as pd



df4 = pd.read_csv('/Users/jkenglish//Desktop/VS_Projects/Streamlit/prediction_df.csv')

X = df4[["COUNTY/STATE","EVENT_TYPE","MONTH_NAME","YEAR"]]
 


y = df4['DAMAGE_PROPERTY']


lasso_reg = Lasso()

lasso_reg.fit(X,y)

joblib.dump(lasso_reg,"/Users/jkenglish//Desktop/VS_Projects/Streamlit/lasso_reg_mod.pkl")




def show_prediction_page():

   #st.title("Prediction Page")
  
  #<div style="background-color: blue ;padding:6px">
   
   predict_html = """
    <div style="background-color: blue ;padding:6px">
    <h2 style="color:white;text-align:center;"> Prediction Page </h2>
    </div>


    """

   st.markdown(predict_html,unsafe_allow_html=True)
    
   
   
   st.write("### Information for Forecasting Property Damages")
   
   
   #state = st.selectbox("Select a state",('NEW JERSEY','TEXAS','NEW YORK','COLORADO', 'ILLINOIS', 'PENNSYLVANIA', 'MICHIGAN', 'WYOMING','FLORIDA','GEORGIA','CALIFORNIA'))

   county_state = st.selectbox("Select a county and state",('WASHINGTON , ARKANSAS','FLAGLER , FLORIDA','COASTAL GLYNN , GEORGIA','ST. JOHNS , FLORIDA','MITCHELL , IOWA','EASTERN HILLSBOROUGH , NEW HAMPSHIRE','HOWARD , IOWA','MIDLAND , MICHIGAN','WAYNE , MICHIGAN','WARREN , VIRGINIA','GREENE , VIRGINIA',' SOMERSET , NEW JERSEY',' MERCER , NEW JERSEY','SANTA CLARITA VALLEY , CALIFORNIA',' CENTRE , PENNSYLVANIA') )
   
   event = st.selectbox('Select an event ',('Hurricane','Ice Storm','Tornado','Cold/Wind Chill','Lake-Effect Snow',
      'Marine High Wind', 'Heavy Rain', 'Funnel Cloud', 'Rip Current', 'Frost/Freeze', 'Lightning','Blizzard', 'Hail', 'Flood', 'Thunderstorm Wind', 'Drought',
       'High Surf', 'Winter Storm', 'Flash Flood','Strong Wind','Tropical Storm'))

   

   month = st.selectbox("Select a month",('February', 'December', 'March', 'October', 'November', 'January',
       'June', 'May', 'April', 'August', 'July', 'September'))



   year = st.slider('Select a year',2022,2032,2022)

   
   
   cost_button = st.button("Predict Cost in Property Damages")

   button_css = st.markdown("""
   
   <style>

   div.stButton>button:first-child{

    background-color: gold;
    color: ffffff;

   }
   
   </style>
   
   """,unsafe_allow_html=True)
   
   if cost_button:

      label_encoder = LabelEncoder()
      
      df = pd.DataFrame([[county_state,event,month,year]],columns=["COUNTY/STATE","EVENT_TYPE","MONTH_NAME","YEAR"])

      df = df.replace({'COUNTY/STATE': {'WASHINGTON , ARKANSAS':21,'FLAGLER , FLORIDA':22,'COASTAL GLYNN , GEORGIA':23,'ST. JOHNS , FLORIDA':24,'MITCHELL , IOWA':25,'EASTERN HILLSBOROUGH , NEW HAMPSHIRE':26,'HOWARD , IOWA':27,'MIDLAND , MICHIGAN':28,'WAYNE , MICHIGAN':29,'WARREN , VIRGINIA':30,'GREENE , VIRGINIA':31,' SOMERSET , NEW JERSEY':32,' MERCER , NEW JERSEY':33,'SANTA CLARITA VALLEY , CALIFORNIA':34,' CENTRE , PENNSYLVANIA':35},
      'EVENT_TYPE': {'Hurricane':5,'Ice Storm':6,'Tornado':7,
       'Cold/Wind Chill':30, 'Lake-Effect Snow':50, 'Ice Storm':90,
      'Marine High Wind':22, 'Heavy Rain':21, 'Funnel Cloud':23, 
      'Rip Current':32, 'Frost/Freeze':33, 'Lightning':42, 'Blizzard':45, 'Hail':29, 'Flood':17, 'Thunderstorm Wind':18, 'Drought':19,
       'High Surf':58, 'Winter Storm':59, 'Flash Flood':60,'Strong Wind':62,'Tropical Storm':75},
      'MONTH_NAME':{'February':8,'December':9,'March':10,'October':11, 
      'November':13, 'January':12, 'June':15, 
      'May':14, 'April':5, 'August':25, 'July':30, 'September':3}})
      
      
      lasso_reg_mod = joblib.load("/Users/jkenglish//Desktop/VS_Projects/Streamlit/lasso_reg_mod.pkl")
      

      
      predict_lasso = lasso_reg_mod.predict(df)

      st.subheader(f"Estimated cost in property damages: ${predict_lasso[0]:,.2f} ")

      
show_prediction_page()
  
      

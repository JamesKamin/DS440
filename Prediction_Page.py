
import streamlit as st

from sklearn.linear_model import LinearRegression

import joblib 

import pandas as pd



df4 = pd.read_csv('/Users/jkenglish//Desktop/VS_Projects/Streamlit/df4.csv')

X = df4[["STATE","EVENT_TYPE","MONTH_NAME","YEAR"]]
 


y = df4['DAMAGE_PROPERTY']


linear_reg = LinearRegression()

linear_reg.fit(X,y)

joblib.dump(linear_reg,"/Users/jkenglish//Desktop/VS_Projects/Streamlit/reg_mod.pkl")




def show_prediction_page():

   st.title("Prediction Page")
  
   st.write("### Information for Forecasting Property Damages")

   
   """state = st.selectbox("Select a state",('NEW HAMPSHIRE', 'MISSOURI', 'KANSAS', 'TEXAS', 'HAWAII',
       'COLORADO', 'ILLINOIS', 'MONTANA', 'MICHIGAN', 'WYOMING',
       'MARYLAND', 'VIRGINIA', 'IOWA', 'NEBRASKA', 'LAKE ST CLAIR',
       'LAKE HURON', 'DISTRICT OF COLUMBIA', 'NEW YORK', 'PENNSYLVANIA',
       'ATLANTIC NORTH', 'GULF OF MEXICO', 'NEW MEXICO', 'OKLAHOMA',
       'WEST VIRGINIA', 'WISCONSIN', 'MINNESOTA', 'ARKANSAS',
       'MISSISSIPPI', 'GEORGIA', 'TENNESSEE', 'LAKE SUPERIOR',
       'RHODE ISLAND', 'MASSACHUSETTS', 'FLORIDA', 'ALABAMA', 'MAINE',
      'ARIZONA', 'OHIO', 'VERMONT', 'LOUISIANA', 'INDIANA',
       'NORTH CAROLINA', 'SOUTH DAKOTA', 'CALIFORNIA', 'UTAH',
       'NORTH DAKOTA', 'KENTUCKY', 'ATLANTIC SOUTH', 'IDAHO',
       'CONNECTICUT', 'LAKE MICHIGAN', 'OREGON', 'ALASKA',
       'SOUTH CAROLINA', 'AMERICAN SAMOA', 'NEVADA', 'WASHINGTON', 'GUAM',
       'HAWAII WATERS', 'PUERTO RICO', 'VIRGIN ISLANDS', 'NEW JERSEY',
       'LAKE ERIE', 'E PACIFIC', 'DELAWARE', 'LAKE ONTARIO',
       'GULF OF ALASKA', 'ST LAWRENCE R'))"""

   state = st.selectbox("Select a state",('NEW JERSEY','TEXAS','NEW YORK','COLORADO', 'ILLINOIS', 'MONTANA', 'MICHIGAN', 'WYOMING'))

   event = st.selectbox('Select an event ',('Hurricane','Ice Storm','Tornado','Cold/Wind Chill','Lake-Effect Snow',
      'Marine High Wind', 'Heavy Rain', 'Funnel Cloud', 'Rip Current', 'Frost/Freeze', 'Lightning','Blizzard', 'Hail', 'Flood', 'Thunderstorm Wind', 'Drought',
       'High Surf', 'Winter Storm', 'Flash Flood'))

   #month = st.selectbox('Select a month',('February','March','December'))

   """event = st.selectbox("Select an event",('Winter Weather', 'Heavy Snow', 'Strong Wind', 'High Wind',
       'Blizzard', 'Hail', 'Flood', 'Thunderstorm Wind', 'Drought',
       'High Surf', 'Winter Storm', 'Flash Flood', 'Tornado', 'Dense Fog',
       'Marine Thunderstorm Wind', 'Debris Flow', 'Excessive Heat',
       'Cold/Wind Chill', 'Lake-Effect Snow', 'Ice Storm',
      'Marine High Wind', 'Heavy Rain', 'Funnel Cloud', 'Rip Current',
       'Waterspout', 'Frost/Freeze', 'Wildfire', 'Lightning',
       'Marine Dense Fog', 'Marine Strong Wind', 'Hurricane',
       'Freezing Fog', 'Astronomical Low Tide', 'Sleet', 'Tsunami',
       'Sneakerwave', 'Storm Surge/Tide', 'Dense Smoke', 'Seiche',
       'Volcanic Ashfall', 'Landslide', 'Tropical Depression',
       'Hurricane (Typhoon)', 'Marine Tropical Storm',
       'Marine Hurricane/Typhoon', 'HAIL FLOODING',
       'THUNDERSTORM WINDS/FLASH FLOOD', 'THUNDERSTORM WINDS LIGHTNING',
       'THUNDERSTORM WIND/ TREES', 'THUNDERSTORM WIND/ TREE',
       'THUNDERSTORM WINDS FUNNEL CLOU', 'TORNADO/WATERSPOUT',
       'THUNDERSTORM WINDS/HEAVY RAIN', 'THUNDERSTORM WINDS HEAVY RAIN',
       'THUNDERSTORM WINDS/ FLOOD', 'Lakeshore Flood',
       'Marine Tropical Depression', 'TORNADOES, TSTM WIND, HAIL',
       'THUNDERSTORM WINDS/FLOODING', 'HAIL/ICY ROADS',
       'Marine Lightning'))"""

   month = st.selectbox("Select a month",('February', 'December', 'March', 'October', 'November', 'January',
       'June', 'May', 'April', 'August', 'July', 'September'))



   year = st.slider('Select a year',2022,2032,2022)
   #year = st.selectbox("Select a year",(2022, 2023, 2024,2025,2026,2027,2028,2029))
   cost_button = st.button("Estimate Cost In Property Damages")
   
   if cost_button:
      
      

      reg_mod = joblib.load("/Users/jkenglish//Desktop/VS_Projects/Streamlit/reg_mod.pkl")

      
      X = pd.DataFrame([[state,event,month,year]],columns=["STATE","EVENT_TYPE","MONTH_NAME","YEAR"])

      
      X = X.replace({'STATE': {'NEW JERSEY': 3, 'TEXAS': 2,
      'NEW YORK':1,'COLORADO':20, 'ILLINOIS':40, 
      'MONTANA':60, 'MICHIGAN':80, 'WYOMING':18},
      'EVENT_TYPE': {'Hurricane':5,'Ice Storm':6,'Tornado':7,
       'Cold/Wind Chill':30, 'Lake-Effect Snow':50, 'Ice Storm':90,
      'Marine High Wind':22, 'Heavy Rain':21, 'Funnel Cloud':23, 
      'Rip Current':32, 'Frost/Freeze':33, 'Lightning':42, 'Blizzard':45, 'Hail':29, 'Flood':17, 'Thunderstorm Wind':18, 'Drought':19,
       'High Surf':58, 'Winter Storm':59, 'Flash Flood':60},
      'MONTH_NAME':{'February':8,'December':9,'March':10,'October':11, 
      'November':13, 'January':12, 'June':15, 
      'May':14, 'April':5, 'August':25, 'July':30, 'September':3}})


      
     
      #X['STATE'] = label_encoder.fit_transform(X['STATE'])

      #X['EVENT_TYPE'] = label_encoder.fit_transform(X['EVENT_TYPE'])

      #X['MONTH_NAME'] = label_encoder.fit_transform(X['MONTH_NAME'])

      

      
      predict_linear = reg_mod.predict(X)

      st.subheader(f"Estimated cost in property damages: ${predict_linear[0]:,.2f} ")

      
      
      

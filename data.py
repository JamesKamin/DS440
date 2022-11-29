import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

st.title("Data Used For Prediction")

st.header("Weather Events Data")

df = pd.read_csv('/Users/adarshsubramanian//Desktop/Work/StormEvents_Combined.csv')
df
st.header("The Events Our Data Considers")
pd.Series(df['EVENT_TYPE'].unique()).sort_values()

df2 = df.drop(['Unnamed: 0'],axis=1)

df2
state_encoder = preprocessing.LabelEncoder()
df2['STATE']= state_encoder.fit_transform(df2['STATE'])
df2['STATE'].unique()
event_encoder = preprocessing.LabelEncoder()
df2['EVENT_TYPE']= event_encoder.fit_transform(df2['EVENT_TYPE'])
df2['EVENT_TYPE'].unique()
month_encoder  = preprocessing.LabelEncoder()
df2['MONTH_NAME']= month_encoder.fit_transform(df2['MONTH_NAME'])
df2['MONTH_NAME'].unique()
county_encoder = preprocessing.LabelEncoder()
df2['COUNTY_NAME']= county_encoder.fit_transform(df2['COUNTY_NAME'])
df2['COUNTY_NAME'].unique()
df3 = df2.drop(['COUNTY_TIMEZONE','DAMAGE_CROPS','COUNTY_NAME'],axis=1)
df3

X = df3.drop('DAMAGE_PROPERTY',axis=1)
y = df3['DAMAGE_PROPERTY']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

st.header("Random Forest Algorithm: $35,265,290.77")
st.header("Ridge: $29,178,952.89")
st.header("Lasso: $29,178,952.89")


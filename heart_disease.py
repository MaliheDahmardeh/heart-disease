import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC ,SVR
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import streamlit as st

#cd C:\Users\Malihe\source\repos\heart disease
# streamlit run heart_disease.py

url = ' https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease?select=heart_2020_cleaned.csv ‘
heart_disease_df _static = get_data(url)
heart_disease_df = heart_disease_df _static.copy()

st.write("let see the Data")
st.header('Heart Disease preciction')
st.write("__Let's predict heart disease")
st.write('Dataset source: [click link](' + url + ')')
st.download_button('DOWNLOAD RAW DATA', get_downloadable_data(heart_disease_static), file_name='heart_disease_raw.csv')

st.sidebar.subheader('Controls')
show_raw_data = st.sidebar.checkbox('Show raw data')

if show_raw_data:
    st.subheader('Raw data')
    st.write(heart_disease_df)


import numpy as np
import pandas as pd
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC ,SVR
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas.core.algorithms import mode
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import streamlit as st

#cd C:\Users\Malihe\source\repos\heart disease
#streamlit run heart_disease.py
#git status
#git commit -m"Adding Codes"
#git push

heart_disease_df =pd.read_csv('https://raw.githubusercontent.com/MaliheDahmardeh/project/main/heart_2020_cleaned.csv')
heart_disease_df.info()

#header

st.header('Heart Disease Analysis')
st.write("Programming Project-University of Verona")
st.write("The dataset come from the CDC and is a major part of the Behavioral Risk Factor Surveillance System , which conducts annual telephone surveys to gather data on the health status of U.S. residents.")
st.write("As the CDC describes Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world. The most recent dataset ,as of February 15, 2022,includes data from 2020.")
st.write(" The vast majority of columns are questions asked to respondents about their health status, such as ,Do you have serious difficulty walking or climbing stairs? or Have you smoked at least 100 cigarettes in your entire life?. In this dataset, many different factors that directly or indirectly influence heart disease.")

if st.checkbox('show raw data'):
     #st.sidebar.subheader('Controls')
     #show_raw_data = st.sidebar.checkbox('Show raw data')
     #show dataset
     #show_raw_data:
   st.subheader('Raw data')
   st.write(heart_disease_df)
if st.checkbox('Data Description'):
   st.write('*Demographic factors: sex, age category, race, BMI') 
   st.write('*Diseases:(weather respondent ever had such diseases) asthma, skin cancer, diabetes, and stroke or kidney disease')
   st.write('*Unhealthy habits:') 
   st.write('Smoking(respondents that smoked at least 100 cigarettes in their entire life)')
   st.write('Alcohol Drinking(heavy drinkers, adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)')
   st.write('*General Health in the last 30 days:')
   st.write('Difficulty Walking (weather respondent have serious difficulty walking or climbing stairs)')
   st.write('Physical Activity (adults who reported doing physical activity or exercise during the past 30 days other than their regular job)')
   st.write('Sleep Time (respondent has reported average hours of sleep in a 24-hour period)')
   st.write('Physical Health (number of days being physically ill or injured)')
   st.write('Mental Health (number of days having bad mental health)')
   st.write('General Health (respondents declared their health:Excellent, Very good, Good, Fair or Poor)')

#data cleaning

heart_disease_df.isnull()
heart_disease_df.isnull().sum().sort_values(ascending=False)
heart_disease_df.nunique()
heart_disease_df.dropna()
heart_disease_df.fillna('non values', inplace =True)
heart_disease_df.drop_duplicates()
heart_disease_df.reset_index()

#data visualization

st.subheader('Data Visualization')

#piecharts
col_1, col_2= st.columns(2)

with col_1:  

     fig= plt.figure(figsize=(10,4))
     heart_disease_df['HeartDisease'].value_counts().plot.pie(explode=[0,0.2],autopct='%.2f', shadow =True, colors=('green','orange'))
     st.write(fig)
     st.caption('percentage of heart disease')
     

with col_2:

     fig= plt.figure(figsize=(10,4))
     heart_disease_df['Sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%.2f', shadow =True , colors=('pink','lightblue'))
     st.write(fig)
     st.caption('Male & Female Distribution')

#Barcharts

#Asthma
fig= plt.figure(figsize=(8,6))
sb.countplot(data=heart_disease_df, x='Asthma', hue='HeartDisease', palette='bwr')
plt.title('Heart Disease distribution and Asthma', fontsize = 14)
plt.xlabel('Asthma')
plt.ylabel('Cases')
st.write(fig)
st.caption('Heart Disease distribution based on Asthma')
if st.checkbox('Asthma and Heart disease details'):
   st.write('Athma, Heart Disease, Cases')
   st.write(heart_disease_df.groupby('Asthma')['HeartDisease'].value_counts())

#Smoking
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(heart_disease_df[heart_disease_df["HeartDisease"]=="No"]["Smoking"], bins=3, alpha=0.8, color="green", label="No HeartDisease")
ax.hist(heart_disease_df[heart_disease_df["HeartDisease"]=="Yes"]["Smoking"], bins=3, alpha=1, color="red", label="HeartDisease")
ax.set_xlabel("Smoking", fontsize = 12)
ax.set_ylabel("cases", fontsize = 12)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=1.)
st.write(fig)
st.caption('heart disease distribution based on Smoking')
if st.checkbox('Smoking and Heart disease details'):
   st.write('Smoking, Heart Disease, Cases')
   st.write(heart_disease_df.groupby('Asthma')['HeartDisease'].value_counts())



#Distribution of BMI
fig, ax = plt.subplots(figsize = (13,5))

sb.kdeplot(heart_disease_df[heart_disease_df["HeartDisease"]=="Yes"]["BMI"], alpha=0.5,shade = True, color="red", label="HeartDisease", ax = ax)
plt.title('Distribution of BMI', fontsize = 14)

ax.set_xlabel("BMI")
ax.set_ylabel("Frequency")
st.write(fig)
st.caption('Distribution of BMI')

if st.checkbox('BMI and Age_Category'):
   st.write('Age Category')
   st.write(heart_disease_df.groupby('AgeCategory')['BMI'].aggregate(['min', np.median, max]))
   st.write(heart_disease_df.describe().T)


#heart disease distribution based on AgeCategory
fig=plt.figure(figsize=(14,7))
sb.countplot(data=heart_disease_df, x='AgeCategory', hue='HeartDisease', order=['18-24', '25-29', '30-34', '35-39', '40-44',
                                                                      '45-49', '50-54', '55-59', '60-64', '65-69',
                                                                      '70-74', '75-79', '80 or older'])

plt.xlabel('Age Category')
plt.ylabel('Cases')
st.write(fig)
st.caption('heart disease distribution based on AgeCategory')


#heart disease distribution based on Race

fig= plt.figure(figsize=(6,6))
race_dist = heart_disease_df['Race'].value_counts(["HeartDisease"]=="Yes")
race_dist.plot(kind='pie', autopct='%.2f')
st.write(fig)
st.caption('heart disease distribution based on Race')

if st.checkbox('Race and HeartDisease'):
  st.write(heart_disease_df.groupby('Race')['HeartDisease'].value_counts())


#heart disease distribution based on Gen Health
fig=plt.figure(figsize=(15,7))
sb.countplot(data=heart_disease_df, x='GenHealth', hue='HeartDisease', order=['Excellent', 'Very good', 'Good',
                                                                              'Fair', 'Poor'])
plt.title('heart disease distribution based on Gen Health', fontsize = 14)
plt.xlabel('Gen Health')
plt.ylabel('Cases')
st.write(fig)
st.caption('heart disease distribution based on Gen Health')

#Disease frequency based on Age Category

encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
heart_disease_df['AgeCategory'] = heart_disease_df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
heart_disease_df['AgeCategory'] = heart_disease_df['AgeCategory'].astype('float')

fig, ax = plt.subplots(figsize = (14,6))
sb.kdeplot(heart_disease_df[heart_disease_df["HeartDisease"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="red", label="HeartDisease", ax = ax)
sb.kdeplot(heart_disease_df[heart_disease_df["KidneyDisease"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="blue", label="KidneyDisease", ax = ax)
sb.kdeplot(heart_disease_df[heart_disease_df["Diabetic"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="orange", label="Diabetic", ax = ax)
sb.kdeplot(heart_disease_df[heart_disease_df["SkinCancer"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="green", label="SkinCancer", ax = ax)

ax.set_xlabel("AgeCategory")
ax.set_ylabel("Frequency")

ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
st.write(fig)
st.caption('Disease frequency based on Age Category')




st.write("most important insights:People with Heart Disease are found to have a higher BMI than people with no Heart Disease .Most Heart Disease Patients are White people and also People found to have heart disease, skin cancer ,diabetics & kidney disease are mostly old people .Most heart disease patients smoke.Gen health and Asthma are not very effective in this dataset.")




#endoding
#print('\nCategorical Columns\n')
heart_disease_df.select_dtypes(include=['O']).nunique()

heart_disease_df['Smoking'] = pd.Series(np.where(heart_disease_df['Smoking'] == 'Yes', 1, 0))
heart_disease_df['AlcoholDrinking'] = pd.Series(np.where(heart_disease_df['AlcoholDrinking'] == 'Yes', 1, 0))
heart_disease_df['Stroke'] = pd.Series(np.where(heart_disease_df['Stroke'] == 'Yes', 1, 0))
heart_disease_df['DiffWalking'] = pd.Series(np.where(heart_disease_df['DiffWalking'] == 'Yes', 1, 0))
heart_disease_df['PhysicalActivity'] = pd.Series(np.where(heart_disease_df['PhysicalActivity'] == 'Yes', 1, 0))
heart_disease_df['Asthma'] = pd.Series(np.where(heart_disease_df['Asthma'] == 'Yes', 1, 0))
heart_disease_df['KidneyDisease'] = pd.Series(np.where(heart_disease_df['KidneyDisease'] == 'Yes', 1, 0))
heart_disease_df['SkinCancer'] = pd.Series(np.where(heart_disease_df['SkinCancer'] == 'Yes', 1, 0))
heart_disease_df['HeartDisease'] = pd.Series(np.where(heart_disease_df['HeartDisease'] == 'Yes', 1, 0))

le = LabelEncoder()
heart_disease_df['Sex']=le.fit_transform(heart_disease_df['Sex'])
heart_disease_df['AgeCategory']=le.fit_transform(heart_disease_df['AgeCategory'])
heart_disease_df['Race']=le.fit_transform(heart_disease_df['Race'])
heart_disease_df['Diabetic']=le.fit_transform(heart_disease_df['Diabetic'])
heart_disease_df['GenHealth']=le.fit_transform(heart_disease_df['GenHealth'])

#HISTOGRAM
#numeric_columns = heart_disease_df.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
#st.set_option('deprecation.showPyplotGlobalUse', False)
#if st.sidebar.checkbox('Histogram Plot') :
   #st.sidebar.subheader("Histogram")
   #histogram_slider = st.sidebar.slider(label="Number of Bins",min_value=10, max_value=100, value=40)
   #sb.distplot(heart_disease_df, bins=histogram_slider)
   #st.pyplot()  
#fig, ax=plt.subplots()
#arr = np.random.normal(1, 1, size=100)
#ax.hist(arr, bins = 60, color="DARKORCHID")
#st.pyplot(fig)

#correlation

st.subheader("Correlation Matrix:")
heart_disease_df.corr()
fig=plt.figure(figsize = (20,20))
sb.heatmap(heart_disease_df.corr(), annot = True)
st.write(fig)

st.write("From Correlation Matrix we can see: Alcohol Drinking,Physical Activity,Gen Health and Sleeping time have the negative/low correlation")

#pairplot
#fig=sb.pairplot(heart_disease_df, hue = 'HeartDisease', vars = ['BMI','PhysicalHealth', 'MentalHealth','DiffWalking'] , palette='Dark2' )
#plt.subplots_adjust(top=0.9)
#st.pyplot(fig)
#st.caption('Distributions of higly correlated features over HeartDisease')




#Remove extra columns
heart_disease_df.drop(['AlcoholDrinking','PhysicalActivity','GenHealth'],axis=1,inplace=True)


#prediction
st.subheader("Prediction:")

#select_model = st.selectbox('select model :', ['RandomForestClassifier' ,'GaussiaNNB', 'Descisiontree'] )



#RandomForestClassifier
st. write("Random Forest Classifier Model:")
#if select_model == RandomForestClassifier():
y_heart_disease_df = heart_disease_df['HeartDisease']
x_heart_disease_df = heart_disease_df.drop( 'HeartDisease' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_heart_disease_df , y_heart_disease_df, test_size=0.2, random_state=42)
model= RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
sum(y_pred == y_test) / len(y_pred)
accuracy_score(y_test, y_pred)
st. write("accuracy: 0.90")

if st.checkbox('KFold Result'):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    i = 0
    for train_index, test_index in kf.split(x_heart_disease_df,y_heart_disease_df):
        i += 1
        model = RandomForestClassifier(random_state=42)
        x_train, x_test = x_heart_disease_df.iloc[train_index], x_heart_disease_df.iloc[test_index]
        y_train, y_test = y_heart_disease_df.iloc[train_index], y_heart_disease_df.iloc[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        accuracies.append(accuracy)
        st. write(i, ') accuracy = ', accuracy)
 
#st. write("Mean accuracy:  0.9006644897768569")
      
#st. write('Mean accuracy: ', np.array(accuracies).mean())


 
#if st.sidebar.checkbox("CONFUSION MATRIX HEAT MAP RandomForest") :      
fig=plt.figure(figsize=(7,5))
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sb.heatmap(cf_matrix/np.sum(cf_matrix),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
st.write(fig)



#GaussianNB

st. write("GaussianNB Model:")
st. write("accuracy: 0.85")
 #select_model == GaussianNB():
 
y2_heart_disease_df = heart_disease_df['HeartDisease']
x2_heart_disease_df = heart_disease_df.drop( 'HeartDisease' , axis = 1) 
x2_train, x2_test, y2_train, y2_test = train_test_split(x2_heart_disease_df , y2_heart_disease_df, test_size=0.2, random_state=42)
model=GaussianNB() 
model.fit(x2_train , y2_train)
y2_pred = model.predict(x_test)
sum(y2_pred == y2_test) / len(y2_pred)
accuracy_score(y_test, y_pred)

    
classification_report(y2_test, y2_pred)

#if  st.sidebar.checkbox(model== 'GaussianNB'):
#c_matrix = confusion_matrix(y2_test, y2_pred)
#plt.figure(figsize=(7,5))
#ax = sb.heatmap(c_matrix/np.sum(c_matrix),fmt='.2%', annot=True, cmap='Blues')
#ax.set_xlabel('\nPredicted Values')
#ax.set_ylabel('Actual Values ');
#ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
#ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
#plt.title('confusion_matrix_heatmap_GaussianNB')
#st.write(fig)


#clasification- Decision Tree

st. write("Decision Tree Model:")
st. write("accuracy: 0.87")

y3_heart_disease_df = heart_disease_df['HeartDisease']
x3_heart_disease_df = heart_disease_df.drop( 'HeartDisease' , axis = 1)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3_heart_disease_df , y3_heart_disease_df, test_size=0.2, random_state=42)
model=DecisionTreeClassifier()
model.fit(x3_train , y3_train)
y3_pred = model.predict(x3_test)
sum(y3_pred == y3_test) / len(y3_pred)
accuracy_score(y3_test, y3_pred)

   
classification_report(y3_test, y3_pred)

#if st.sidebar.checkbox(model== 'DecisionTreeClassifier'):
#f_matrix = confusion_matrix(y3_test, y3_pred)
#plt.figure(figsize=(7,5))
#ax = sb.heatmap(cf_matrix/np.sum(f_matrix),fmt='.2%', annot=True, cmap='Blues')
#ax.set_xlabel('\nPredicted Values')
#ax.set_ylabel('Actual Values ');
#ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
#ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
#plt.title('confusion_matrix_heatmap_DecisionTreeClassifier')
#st.write(fig)
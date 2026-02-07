import streamlit as st
import pandas as pd
import joblib
st.title("❤️ Cardiac Disease Prediction")
Age=st.number_input('Enter your age',20,90)
gender= st.radio("Select your gender", ["Male", "Female"])
if gender =='Male':
    gender=1
else:
    gender=0    
cholestrol=st.number_input('Enter cholestrol value',50,500)  
BP=st.number_input('Enter your blood pressure',70,280)
HR=st.number_input('Enter your Heart Rate',40,150)
smoking=st.radio('Select your smoking status',['Current','Never','Former'])
if smoking=='Current':
    smoking=0
elif smoking=='Never':
    smoking=2
else:
    smoking=1        
alchol=st.radio('Alchol intake :',['Heavy','Moderate'])
if alchol=='Heavy':
    alchol=0
else:
    alchol=1
excersie=st.number_input('Enter excersie hours:',0,10)  
family_hist=st.radio('Do you have any family history of cardiac disease',['Yes','No'])
if family_hist =='Yes':
    family_hist=1
else:
    family_hist=0
diabetes=st.radio('Do you have Diabetes',['Yes','No'])
if diabetes=='Yes':
    diabetes=1
else:
    diabetes=0  
obesity=st.radio('Do you have obesity',['Yes','No'])
if obesity=='Yes':
    obesity=1
else:
    obesity=0                    
agina=st.radio('Select Exercise induced Agina',['Yes','No'])
if agina=='Yes':
    agina=1
else:
    agina=0
stress=st.number_input('Enter your stress level',2,10)  
blood_sugar=st.number_input('Enter your blood sugar level',50,250)
chest_pain=st.radio('Select chest pain type :',['Atypical Angina' ,'Typical Angina', 'Non-anginal Pain' ,'Asymptomatic'])               
if chest_pain=='Atypical Angina':
    chest_pain=1
elif chest_pain=='Typical Angina':
    chest_pain=3
elif chest_pain=='Non-anginal Pain':
    chest_pain=2
else:
    chest_pain=0
frame=pd.DataFrame({
    'Age':[Age],
    'Gender':[gender],
    'Cholesterol':[cholestrol],
    'Blood Pressure':[BP],
    'Heart Rate':[HR],
    'Smoking':[smoking],
    'Alcohol Intake':[alchol],
    'Exercise Hours':[excersie],
    'Family History':[family_hist],
    'Diabetes':[diabetes],
    'Obesity':[obesity],
    'Stress Level':[stress],
    'Blood Sugar':[blood_sugar],
    'Exercise Induced Angina':[agina],
    'Chest Pain Type':[chest_pain]
    }) 
st.subheader('Click here to verify your data ') 
if st.button('Click here',key='buttoon1') : 
    st.subheader('Your data shown below')
    st.dataframe(frame)             
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
model, scaler = load_model()
st.subheader('Click here to predict heart disease')
if st.button("🔍 Predict Heart Disease",key="button2"):
    scaled_input = scaler.transform(frame)
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]
    if prediction[0] == 1:
        st.error("⚠️ High risk of cardiac disease")
    else:
        st.success("✅ Low risk of cardiac disease")

    st.write(f"Risk Probability:{prob:.2f}")
chart_data = pd.DataFrame({
    "Values": [
        cholestrol,
        BP,
        HR,
        stress,
        blood_sugar
        ]
    }, index=[
        "Cholesterol",
        "Blood Pressure",
        "Heart Rate",
        "Stress Level",
        "Blood Sugar"
    ])
st.subheader("📊Health Metrics Overview")
st.bar_chart(chart_data)
    

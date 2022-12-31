import streamlit as st

import joblib
import os
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import numpy as np
import pandas as pd

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

class File_downloader(object):
    """doctstring for file_downloader
    >>>download = file_download(data,filename,file_ext='csv')

    """
    def __init__(self,data,filename = 'myfile',file_ext='csv'):
        super(File_downloader,self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext
    
    def download(self):
    
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
        st.markdown("####Download File ###")
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
        st.markdown(href,unsafe_allow_html=True)

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_ml_app():
    st.title("From ML section")
    #st.write("It is working")
    #st.success("Wow it is so cool")
    submenu = st.sidebar.radio("Submenu",["Single-Online Prediction","Batch Predictions"])
    if submenu == "Single-Online Prediction":
        st.subheader("Single-Online Prediction")
        col1,col2 = st.columns(2)



        with col1:
            
            pregnancies = st.number_input("Pregnancies",0,20)
            glucose = st.number_input("Glucose",0,200)
            bloodpressure = st.number_input("BloodPressure",0,130)
            skinthickness = st.number_input("SkinThickness",0,100)
            #gender = st.radio("Gender",("Female","Male"))

        with col2:
            insulin = st.number_input("Insulin",0,850)
            bmi = st.number_input("BMI",0,70)
            dpf = st.number_input("DiabetesPedigreeFunction",0.05,2.5)
            age = st.number_input("Age",20,100)

        with st.expander("Your Selected options:"):
            result = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': bloodpressure,
                'SkinThickness':skinthickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreefunction':dpf,
                'Age': age

            }

            st.write(result)

            encoded_result = []
            for i in result.values():
                if type(i) == int:
                    encoded_result.append(i)
                else:
                    encoded_result.append(i)
        

            st.write(encoded_result)

        with st.expander("Prediction Result"):
            single_sample = np.array(encoded_result).reshape(1,-1)
            st.write(single_sample)

            model = load_model("model_rf.pkl")
            prediction = model.predict(single_sample)
            pred_prob = model.predict_proba(single_sample)
            st.write(prediction)
            st.write(pred_prob)


            
            

    elif submenu == "Batch Predictions":
        st.subheader("Batch Predictions ")
        with st.expander("Batch Predictions upload data"):
            data_file = st.file_uploader("Upload CSV",type="csv")
            if data_file is not None:
                file_details = {"filename":data_file.name,
                "filetype":data_file.type,"filesize":data_file.size}
                st.write(data_file)
                
                test = pd.read_csv(data_file)
                # test = load_data("test_data.csv")
                test_result = test.copy()
                st.dataframe(test)

            submenu = st.sidebar.selectbox("Models",["Default","RandomForestClassifier","LogisticRegression","DecisionTreeClassifier"])
        with st.expander("Batch Predictions"):

            if submenu == "Default":
                st.subheader("Choose your model")
                st.markdown("Upload your file and choose a model to carry out your predictions")
            
            elif submenu == "RandomForestClassifier":
                st.subheader("RandomForestClassifier")
                model = load_model("model_rf.pkl")
                prediction_batch_rf = model.predict(test)
                pred_prob_batch_rf = model.predict_proba(test)
                st.write(prediction_batch_rf)
                st.write(pred_prob_batch_rf)
                test_result['Predict_Outcome_RandomForest' ] = prediction_batch_rf
                test_result[["Probability Outcome = 0","Predicted Outcome = 1"]] = pred_prob_batch_rf
                download = File_downloader(test_result.to_csv()).download()

            elif submenu == "LogisticRegression":
                st.subheader("LogisticRegression")
                model = load_model("model_lr.pkl")
                prediction_batch_lr = model.predict(test)
                pred_prob_batch_lr = model.predict_proba(test)
                st.write(prediction_batch_lr)
                st.write(pred_prob_batch_lr)
                test_result['Predict_Outcome_LogisticRegression' ] = prediction_batch_lr
                test_result[["Probability Outcome = 0","Predicted Outcome = 1"]] = pred_prob_batch_lr
                download = File_downloader(test_result.to_csv()).download()

            else:
                st.subheader("DecisionTreeClassifier")
                model = load_model("model_df.pkl")
                prediction_batch_dt = model.predict(test)
                pred_prob_batch_dt = model.predict_proba(test)
                st.write(prediction_batch_dt)
                st.write(pred_prob_batch_dt)
                test_result['Predict_Outcome_DecisionTree' ] = prediction_batch_dt
                test_result[["Probability Outcome = 0","Predicted Outcome = 1"]] = pred_prob_batch_dt
                
                download = File_downloader(test_result.to_csv()).download()

                #test['Predicted_Outcome'] = prediction_batch
                #test['Predicted_Probability'] = pred_prob_batch

                #test_results = test.copy()

                # results = pd.concat([test, prediction_batch,pred_prob_batch], axis=1)
                #st.write(test_results)
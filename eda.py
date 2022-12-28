import streamlit as st

import pandas as pd

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("From Exploratory Data Analysis ")
    df = load_data("diabetes.csv")
    df_freq = load_data("freqdist_of_age_data.csv")
    

    submenu = st.sidebar.selectbox("Submenu",["Descriptive","Plots"])
    if submenu == "Descriptive":
        if st.button("Display_Data"):
            st.dataframe(df)

            with st.expander("Data Types"):
                st.dataframe(df.dtypes)

            with st.expander("Descriptive Summary"):
                st.dataframe(df.describe())
            with st.expander("Class Distribution"):
                st.dataframe(df['Outcome'].value_counts())
        

    elif submenu == "Plots":
        st.subheader("Plots")
        col1,col2 = st.columns([2,1])
        
        with col1:
            with st.expander("Distribution Plot of Gender"):
                fig = plt.figure()
                sns.countplot(df['Pregnancies'])
                st.pyplot(fig)

                preg_df = df['Pregnancies'].value_counts().to_frame()
                preg_df = preg_df.reset_index()
                preg_df.columns=["No of Pregnancies","Count"]
                # st.dataframe(preg_df)

                p1 = px.pie(preg_df,names = 'No of Pregnancies',values = 'Count')
                st.plotly_chart(p1,use_container_width=True)
        
            with st.expander("Distribution of Outcomes"):
                fig = plt.figure()
                sns.countplot(df['Outcome'])
                st.pyplot(fig)

        with col2:
            with st.expander("Pregnancy Distribution"):
                st.dataframe(preg_df)

            with st.expander("Disribution of Outcomes"):
                st.dataframe(df['Outcome'].value_counts())
        
        with st.expander("Distribution of Age Frequency"):
            #st.dataframe(df_freq)
            p2 = px.bar(df_freq,x = 'Age',y = 'count')
            st.plotly_chart(p2)

        with st.expander("Outlier Detection"):
            fig = plt.figure()
            sns.boxplot(df['Age'])
            st.pyplot(fig)

            p3 = px.box(df,x='Age',color = 'Outcome')
            st.plotly_chart(p3)

        with st.expander("Correlation Plot"):
            corr_matrix = df.corr()
            fig = plt.figure(figsize = (20,10))
            sns.heatmap(corr_matrix,annot=True)
            st.pyplot(fig)

            # p4 = px.imshow(corr_matrix)
            # st.plotly_chartp4
        


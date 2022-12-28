import streamlit as st
import streamlit.components.v1 as stc

from eda import run_eda_app
from ml import run_ml_app

html_temp = """
    <div style="background-color:#3872fb;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Early Stage DM Risk Data App </h1>
    <h4 style="color:white;text-align:center;">Diabetes </h4>
    </div>
    """


def main():
    #st.title("Main App")
    stc.html(html_temp)

    menu = ["Home","EDA","ML","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        st.subheader("About")
        
if __name__ == '__main__':
    main()
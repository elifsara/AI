import streamlit as st
from pycaret.classicication import setup, compare_models, pull, save_modele, load_model
import pandas as pd 
from streamlit_pandas_profiling import st_profile_report
#import ydata_profiling
import os

if os.path.exists('./dataset.csv'):
    df=pd.read_csv('dataset.csv',index_col=None)

with st.sidebar:
    st.image('https://cdn0.iconfinder.com/data/icons/machine-learning-flat/60/021_Decision_Making-1024.png')
    st.title('AutoML Classifiction')
    choice=st.radio('Navigation',['Upload','Modelling','Download'])

if choice=='Upload':
    st.title('Upload Your Data')
    file=st.file_uploader('Upload Your Dataset')
    if file:
        df=pd.read_csv(file,index_col=None)

        df.to_csv('dataset.csv', index=None)
        st.dataframe(df.head())

#if choice=='EDA':
#   st.title('Exploratory Data Analysis')
#    pofile_df=df.profile_report()
#    st_profile_report(profile_df)
    


if choice=='Modelling':
    target=st.selectbox('Choose the target column', df.columns)
    if st.button('train model'):
        setup(df,target=target)
        setup_df=pull()
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.dataframe(compare_df)
        save_model(best_model,'best_model')

if choice=='Download':
    with open('best_model.pkl','rb') as f:
        st.download('Download Best modeel',f,'best_model.pkl')


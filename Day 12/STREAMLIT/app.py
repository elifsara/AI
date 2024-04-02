import streamlit as st
import pickle
st.title('Tecrube, yazili ve sozlu sinava gore maas tahmin eden model :heavy_dollar_sign:')
model=pickle.load(open('maas.pkl','rb'))
tecrube=st.number_input('Tecrube',1,10)
yazili=st.number_input('yazili',1,10)
mulakat=st.number_input('mulakat',1,10)
if st.button('Tahmin et'):
    tahmin=model.predict([[tecrube,yazili,mulakat]])
    tahmin=round(tahmin[0],2)
    st.success(f'maas tahmini: {tahmin}')
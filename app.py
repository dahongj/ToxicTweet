import numpy as np
import streamlit as st
from transformers import pipeline

def bertweet(data):
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    print(specific_model(data))

def getSent(data, model):
    if(model == 'Bertweet'):
        bertweet(data)


def rendPage():
    st.title("Sentiment Analysis")
    userText = st.text_input('User Input', "Hope you are having a great day!")
    st.text("")
    type = st.selectbox(
        'Choose your model',
        ('Bertweet','Roberta'))
    st.text("")

    if st.button('Calculate'):
        if(userText!="" and type != None):
            st.text
            getSent(userText,type)

rendPage()

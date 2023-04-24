import numpy as np
import streamlit as st
from transformers import pipeline
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

import torch

def bertweet(data):
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    result = specific_model(data)
    label = result[0]['label']
    score = result[0]['score']

    return label, score 

def roberta(data):
    specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    result = specific_model(data)
    label = result[0]['label']
    score = result[0]['score']

    if(label == 'LABEL_0'):
        label = 'Negative'
    elif(label == 'LABEL_1'):
        label = 'Neutral'
    else:
        label = 'Positive'

    return label, score 

def siebert(data):
    specific_model = pipeline(model='siebert/sentiment-roberta-large-english')
    result = specific_model(data)
    label = result[0]['label']
    score = result[0]['score']

    return label, score

def finetuned(data):
    model_name = "dahongj/finetuned_toxictweets"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenized_text = tokenizer(data, return_tensors="pt")
    res = model(**tokenized_text)
    mes = torch.sigmoid(res.logits)

    Dict = {0: "toxic", 1: "severe_toxic", 2: "obscene", 3: "threat", 4: "insult", 5: "identity_hate"}

    maxres, maxscore, sec, secscore = Dict[0], mes[0][0].item(), 0, 0

    for i in range(1,6):
        if mes[0][i].item() > secscore:
            sec = i
            secscore = mes[0][i].item()

    return maxres, maxscore, Dict[sec], secscore

def getSent(data, model):
    if(model == 'Bertweet'):
        label,score = bertweet(data)
        col1, col2 = st.columns(2)
        col1.metric("Feeling",label,None)
        col2.metric("Score",score,None)
    elif(model == 'Roberta'):
        label,score = roberta(data)
        col1, col2 = st.columns(2)
        col1.metric("Feeling",label,None)
        col2.metric("Score",score,None)
    elif(model == 'Siebert'):
        label,score = siebert(data)
        col1, col2 = st.columns(2)
        col1.metric("Feeling",label,None)
        col2.metric("Score",score,None)
    elif(model == 'Finetuned'):
        label, score, sec, secsc = finetuned(data)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Highest",label,None)
        col2.metric("Score",score,None)
        col3.metric("Second Highest", sec, None)
        col4.metric("Score", secsc, None)

def rendPage():
    st.title("Sentiment Analysis")
    userText = st.text_area('User Input', "Hope you are having a great day!")
    st.text("")
    type = st.selectbox(
        'Choose your model',
        ('Bertweet','Roberta','Siebert','Finetuned'))
    st.text("")

    if st.button('Calculate'):
        if(userText!="" and type != None):
            st.text("")
            getSent(userText,type)

rendPage()

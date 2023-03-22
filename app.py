import tensorflow 
from tensorflow import keras
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import spacy
from transformers import pipeline
from datasets import load_dataset
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

with st.sidebar:
    selected=option_menu('NLP on the GO by Rishabh Patil',
                         ['Tokens and Lemma',
                          'Named Entity','Sentiment Analysis','Text Summarization'],
                          icons=['coin','code','emoji-smile','card-text'],
                          default_index=0)

if selected=='Tokens and Lemma':
    def text_analyzer(my_text):
        nlp=spacy.load('en_core_web_sm')
        docx=nlp(my_text)
        allData=[("'Tokens': {},\n 'Lemmas': {}".format(token.text,token.lemma_)) for token in docx]
        return allData
        st.title('Tokens and Lemmas')
    
    st.subheader('Tokenize your text')
    message=st.text_area('Enter your Text','Type Here')
    if st.button('Analyze'):
        nlp_result=text_analyzer(message)
        st.json(nlp_result)

if selected=='Named Entity':
    def entity_analyzer(my_text):
        nlp=spacy.load('en_core_web_sm')
        docx=nlp(my_text)
        tokens=[token.text for token in docx]
        entities=[(entity.text,entity.label_)for entity in docx.ents]
        allData=[("'Tokens': {},\n 'Entities': {}".format(tokens,entities))]
        return allData
    st.title('Show Named Entity')
    
    st.subheader('Extract entities from your text')
    message=st.text_area('Enter your Text','Type Here')
    if st.button('Extract'):
        nlp_result=entity_analyzer(message)
        st.json(nlp_result)

if selected=='Sentiment Analysis':
    maxlen=50
    model = keras.models.load_model('tweet_model.h5')

    df = load_dataset("emotion")
    train=df['train']

    def get_tweet(data):
        tweet=[x['text'] for x in data]
        label=[x['label'] for x in data]
        return tweet,label

    tweet,label=get_tweet(train)

    tokenizer=Tokenizer(num_words=1000,oov_token='<UNK>')
    tokenizer.fit_on_texts(tweet)

    def get_sequences(tokenizer,tweet):
        sequences=tokenizer.texts_to_sequences(tweet)
        padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen)
        return padded
    st.title('Show Sentiment Analysis')
    
    st.subheader('Sentiment of your text')
    sentence=st.text_area('Enter your Text','Type Here')
    if st.button('Analyze'):
        sen_seq=get_sequences(tokenizer,[sentence])
        pred=model.predict(sen_seq)[0]
        p=np.argmax(pred)
        emotion=''
        if p==0:
            emotion='sadness'
        elif p==1:
            emotion='joy'
        elif p==2:
            emotion='love'
        elif p==3:
            emotion='anger'
        elif p==4:
            emotion='fear'
        elif p==5:
            emotion='suprise'
        st.success(emotion)

if selected=='Text Summarization':
    summarizer=pipeline('summarization')
    st.title('Show Text Summarization')
    
    st.subheader('Summarize your text')
    maxlen=st.slider('Maximum length of summary',0,200,25)
    minlen=st.slider('Minimum length of summary',0,50,25)
    message=st.text_area('Enter your Text','Type Here')
    if st.button('Summarize'):
        sum=summarizer(message,max_length=maxlen,min_length=minlen,do_sample=False)
        st.success(sum[0]['summary_text'])

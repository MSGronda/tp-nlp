import re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import nltk
import os
from langdetect import detect


def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = text.split()
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(word for word in text)
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def clean_tweets(df):
    ssl._create_default_https_context = ssl._create_unverified_context

    nltk.download('stopwords')
    nltk.download('wordnet')

    # Aplicamos preprocesamiento de tweets

    df['cleantext'] = df['tweet'].apply(clean_text)

    return df

def filter_english(df):

    # Nos quedamos con solo los tweets en ingles

    with ThreadPoolExecutor() as executor:
        df['language'] = list(executor.map(detect_language, df['cleantext']))

    df = df[df['language'] == 'en']

    return df



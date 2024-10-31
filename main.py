import os

import pandas as pd

from graph import graph_distribution
from preprocessing import  clean_tweets, filter_english
from sentiment_analysis import do_analysis, MODEL_BERT, MODEL_DISTILBERT, MODEL_BERTWEET, MODEL_ROBERTA, \
    do_vader_analysis

if __name__ == '__main__':

    dataset = 'dataset/hashtag_donaldtrump.csv'
    clean_text = "./clean_text.csv"

    if not os.path.isfile(clean_text):

        print("Loading dataset...")
        df = pd.read_csv(dataset, engine='python')

        print("Cleaning text...")
        df = clean_tweets(df)

        print("Filtering english tweets")
        df = filter_english(df)
        df.to_csv(clean_text)
    else:
        df = pd.read_csv(clean_text, engine='python')


    print("Starting sentiment analysis")
    do_vader_analysis(df, './vader.csv')
    do_analysis(df, MODEL_DISTILBERT, './distilbert.csv')
    do_analysis(df, MODEL_BERTWEET, './bertweet.csv')
    do_analysis(df, MODEL_ROBERTA, './roberta.csv')
    print("Sentiment analysis complete")



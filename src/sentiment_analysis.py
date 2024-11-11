import pandas as pd
from datasets import Dataset
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_BERT = 'nlptown/bert-base-multilingual-uncased-sentiment'
MODEL_DISTILBERT = 'DT12the/distilbert-sentiment-analysis'
MODEL_BERTWEET = 'finiteautomata/bertweet-base-sentiment-analysis'
MODEL_ROBERTA = 'cardiffnlp/twitter-roberta-base-sentiment'

def map_label_roberta(label):
    if label == 'LABEL_0':
        return 'negative'
    elif label == 'LABEL_1':
        return 'neutral'
    elif label == 'LABEL_2':
        return 'positive'

def map_label_bertweet(label):
    if label == 'NEG':
        return 'negative'
    elif label == 'NEU':
        return 'neutral'
    elif label == 'POS':
        return 'positive'

def map_label_distiblert(label):
    if label == 'LABEL_0':
        return 'negative'
    elif label == 'LABEL_1':
        return 'neutral'
    elif label == 'LABEL_2':
        return 'positive'
    raise ValueError(label)

def map_label_bert(label):
    if label  == '1 star' or label == '2 stars':
        return 'negative'
    elif label == '3 stars':
        return 'neutral'
    else:
        return 'positive'


def do_analysis(df, model, output_file_path):
    dataset = Dataset.from_pandas(df[['cleantext']].dropna())

    if model == MODEL_BERT:
        map_function = map_label_bert
    elif model == MODEL_DISTILBERT:
        map_function = map_label_distiblert
    elif model == MODEL_BERTWEET:
        map_function = map_label_bertweet
    elif model == MODEL_ROBERTA:
        map_function = map_label_roberta

    sentiment_pipeline = pipeline('sentiment-analysis', model=model, device=0)

    # Un asco esto
    def analyze_sentiment(batch):
        sentiment_result = sentiment_pipeline(batch['cleantext'])[0]
        batch['sentiment'] = map_function(sentiment_result['label'])
        return batch

    results = dataset.map(analyze_sentiment, batched=False)

    result_df = pd.DataFrame(results)

    result_df.to_csv(output_file_path, index=False)

def do_vader_analysis(df, output_file_path):
    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        return analyzer.polarity_scores(text)
    df['vader_sentiment'] = df['cleantext'].apply(get_vader_sentiment)
    df['vader_compound'] = df['vader_sentiment'].apply(lambda score_dict: score_dict['compound'])

    def get_vader_analysis(compound):
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df['vader_compound'].apply(get_vader_analysis)

    df.drop(columns=['vader_sentiment', 'vader_compound'], inplace=True)

    result_df = df[['cleantext', 'sentiment']]

    result_df.to_csv(output_file_path, index=False)

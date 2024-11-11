import re

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as nltkStopwords


df = pd.read_csv('../dataset/hashtag_donaldtrump.csv', engine='python', encoding='ISO-8859-1', on_bad_lines='skip')

text_column = df['tweet'].dropna()

text = ' '.join(text_column)

text = re.sub(r'http\S+|www\S+', '', text)

text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text)

text = text.lower()

stopwords = STOPWORDS
stopwords.update(['de', 'en', 'que', 'se', 'y'])
spanish_stopwords = set(nltkStopwords.words('spanish'))
stopwords.update(spanish_stopwords)

stopwords.update(['trump', 'trumps', 'donald', 'biden', 'joe', 'kamala', 'harris', 'donaldtrump','realdonaldtrump', 'joebiden', 'bidenharris'])

print("Creating WordCloud")
wordcloud = WordCloud(
    width=800,
    height=400,
        stopwords=STOPWORDS,
    background_color='white',
    collocations=False
).generate(text)

# Display the word cloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud.to_file('wordcloud_image_without_names.png')


print("Processing most frequent words")
words = text.split()
filtered_words = [word for word in words  if word  not in stopwords]

word_counts = Counter(filtered_words)

top_20 = word_counts.most_common(20)

words, counts = zip(*top_20)

plt.figure(figsize=(10, 6))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')
plt.gca().invert_yaxis()
plt.show()


print("Processing word appearance percentage")
tweet_count = len(text_column)

word_presence = {}
for word in words:
    word_presence[word] = sum(df['tweet'].str.contains(rf'\b{word}\b', case=False, na=False))

percentages = [(word_presence[word] / tweet_count) * 100 for word in words]

plt.figure(figsize=(10, 6))
plt.barh(words, percentages, color='lightcoral')
plt.xlabel('Porcentaje de aparicion')
plt.ylabel('Palabra')
plt.gca().invert_yaxis()
plt.show()


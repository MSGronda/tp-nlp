import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


# Load your data
data = pd.read_csv('./dataset/bertweet.csv')

# Dependent Variable: log transformation of likes
data['likes'].fillna(0, inplace=True)
data['log_likes'] = np.log(data['likes'] + 1)

# Independent Variables

# 1. Time of Posting and Day of the Week
data['created_at'] = pd.to_datetime(data['created_at'])
data['hour'] = data['created_at'].dt.hour
data['day_of_week'] = data['created_at'].dt.dayofweek

# Cyclical transformation for time-based features
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# 2. Tweet Length
data['tweet_length'] = data['tweet'].str.len()

# 3. Hashtag Count
data['hashtag_count'] = data['tweet'].str.count('#')

# 4. User's Follower Count
data['user_followers_count'] = data['user_followers_count'].fillna(0)

# 5. Sentiment
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['sentiment'] = data['sentiment'].map(sentiment_mapping)

# Bag-of-Words (or TF-IDF) for the most used words in tweets
# Use CountVectorizer for BoW or TfidfVectorizer for TF-IDF
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20
)  # Adjust max_features based on top N words you want


# Transform the 'tweet' column
bow_features = vectorizer.fit_transform(data['tweet'].fillna('')).toarray()

# Create a DataFrame from the BoW features
bow_df = pd.DataFrame(bow_features, columns=vectorizer.get_feature_names_out())

# Concatenate BoW features with the original dataset
data = pd.concat([data, bow_df], axis=1)
print(list(bow_df.columns))
# Define independent variables (X) and dependent variable (y)
X = data[['user_followers_count', 'hashtag_count', 'hour_sin', 'hour_cos',
          'day_sin', 'day_cos', 'sentiment', 'tweet_length'] + list(bow_df.columns)]
y = data['log_likes']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
score = model.score(X_test, y_test)
print("Model R^2 Score:", score)


# Interpret the coefficients for the words
coefficients = model.coef_
word_coefficients = dict(zip(X.columns, coefficients))
sorted_word_impact = sorted(word_coefficients.items(), key=lambda item: item[1], reverse=True)
print("\nTop Words Impacting Virality:")
for word, coef in sorted_word_impact:
    print(f"{word}: {coef:.4f}")


# Actual vs. Predicted Plot
# Predict on the test set
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual log(likes + 1)")
plt.ylabel("Predicted log(likes + 1)")
plt.title("Actual vs. Predicted log(likes + 1)")
plt.show()

# Extract top positive and negative coefficients for BoW features
coefficients = model.coef_[-len(bow_df.columns):]
words = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({'word': words, 'coefficient': coefficients})
top_positive_words = coef_df.nlargest(7, 'coefficient')
top_negative_words = coef_df.nsmallest(7, 'coefficient')

# Plotting top positive and negative word coefficients
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Top Positive Coefficients
ax[0].barh(top_positive_words['word'], top_positive_words['coefficient'], color='green')
ax[0].set_title("Top 5 Positive Word Coefficients")
ax[0].set_xlabel("Coefficient Value")

# Top Negative Coefficients
ax[1].barh(top_negative_words['word'], top_negative_words['coefficient'], color='red')
ax[1].set_title("Top 5 Negative Word Coefficients")
ax[1].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.show()



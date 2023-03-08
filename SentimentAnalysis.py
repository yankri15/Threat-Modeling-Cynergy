import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB

# Load the data
df = pd.read_csv('data.csv')
nltk.download('vader_lexicon')

# Create label dictionary
label_dict = {0: 0, 1: 1}

df['threat'] = df['threat'].apply(lambda x: label_dict[x])

# Define X and y
x = df['sentence']
y = df['threat']

# Split into training and validation sets
train_text, val_text, train_labels, val_labels = train_test_split(x, y, test_size=0.3, random_state=42)

# Bag of Words Vectorization-Based Model
bow_vectorizer = CountVectorizer()
bow_train = bow_vectorizer.fit_transform(train_text)
bow_val = bow_vectorizer.transform(val_text)

# Train model and make predictions
mnb = MultinomialNB()
mnb.fit(bow_train, train_labels)
mnb_predictions = mnb.predict(bow_val)
print('Bag of Words Model Accuracy:', accuracy_score(val_labels, mnb_predictions))

# VADER Model
sid = SentimentIntensityAnalyzer()
vader_predictions = [int(sid.polarity_scores(sentence)['compound'] > 0) for sentence in val_text]
print('VADER Model Accuracy:', accuracy_score(val_labels, vader_predictions))

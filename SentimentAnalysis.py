import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer

def get_max_sentence_length(df):
    max_length = 0
    for sentence in df['sentence']:
        length = len(sentence.split())
        if length > max_length:
            max_length = length
    return max_length


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
train_text, val_text, train_labels, val_labels = train_test_split(x, y, test_size=0.25 , random_state=50)

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
vader_predictions = [int(sid.polarity_scores(sentence)['compound'] >= 0.05) for sentence in val_text]
print('VADER Model Accuracy:', accuracy_score(val_labels, vader_predictions))


# Deep learning model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)

vocab_size = len(tokenizer.word_index) + 1

max_len = get_max_sentence_length(df)

train_sequences = tokenizer.texts_to_sequences(train_text)
val_sequences = tokenizer.texts_to_sequences(val_text)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding='post')

model = Sequential()
max_len = get_max_sentence_length(df)

model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels), epochs=10, batch_size=16)

# Evaluate the model
loss, accuracy = model.evaluate(val_padded, val_labels)
print('Validation accuracy:', accuracy)
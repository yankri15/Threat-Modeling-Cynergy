import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix


import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_bow_model_results(model, val_padded, val_labels):
    # Fit the model with different alpha values and store the results
    alphas = np.arange(0.1, 1.0, 0.1)
    cv_results = []
    for alpha in alphas:
        model.set_params(alpha=alpha)
        scores = cross_val_score(model, val_padded, val_labels, cv=10, scoring='accuracy')
        cv_results.append(scores.mean())

    # Create accuracy graph
    plt.plot(alphas, cv_results)
    plt.title('Bag-of-Words Model Accuracy')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.show()

    # plot confusion matrix
    y_pred = model.predict(val_padded)
    cm = confusion_matrix(val_labels, y_pred)
    plot_confusion_matrix(cm, classes=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.show()


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
train_text, val_text, train_labels, val_labels = train_test_split(x, y, test_size=0.20 , random_state=43)

# Bag of Words Vectorization-Based Model
bow_vectorizer = CountVectorizer()
bow_train = bow_vectorizer.fit_transform(train_text)
bow_val = bow_vectorizer.transform(val_text)

# Train model and make predictions
mnb = MultinomialNB()
mnb.fit(bow_train, train_labels)
mnb_predictions = mnb.predict(bow_val)
print('Bag of Words Model Accuracy:', accuracy_score(val_labels, mnb_predictions))

plot_bow_model_results(mnb, bow_val, val_labels)


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
model.add(LSTM(units=64, dropout=0.15, recurrent_dropout=0.25))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels), epochs=10, batch_size=256)

# Evaluate the model
loss, accuracy = model.evaluate(val_padded, val_labels)
print('Validation accuracy:', accuracy)

# Notes for further steps to to:
# Steps to categorize threats by tags using a DNN model:

# 1. Modify the label_threats function to assign tags to sentences containing threats.
# 2. Update the DataPreProcessing.py file to include the modified label_threats function.

#    - In the label_threats function:
#      - Create a dataframe to store sentences and their corresponding threat tags.
#      - Iterate over each sentence.
#      - Check if the sentence contains any of the threat tags.
#      - Create a new row with the sentence and its threat tags.
#      - Add the new row to the dataframe.
#      - Return the labeled dataframe.

# 3. Train a DNN model using the labeled dataset.
# 4. Use the trained model to classify sentences and assign tags to the threats.

#    - Load the trained DNN model.
#    - For each sentence, pass it through the model and get the predicted tag(s).
#    - Update the dataframe with the predicted tags for the threats.

# 5. Evaluate the performance of the model using appropriate metrics.
# 6. Use the categorized threats for further analysis or decision-making.

# Make sure to save the modified dataframe with the labeled threats and predicted tags for future use.

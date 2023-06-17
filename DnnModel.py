import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers, models

# Load the preprocessed dataset
df = pd.read_csv("data.csv")

# Prepare the input data
X = df["sentence"].values

# Prepare the output data
y_registerForm = df["registerForm"].values
y_rstPss = df["rstPss"].values
y_chatBot = df["chatBot"].values
y_bagCart = df["bagCart"].values
y_accountSettings = df["accountSettings"].values
y_couponPromotion = df["couponPromotion"].values
y_LoginLogout = df["LoginLogout"].values
y_contactForm = df["contactForm"].values
y_searchFeature = df["searchFeature"].values
y_linksPage = df["linksPage"].values
y_fileUpload = df["fileUpload"].values
y_fileDownload = df["fileDownload"].values
y_wishlistFeature = df["wishlistFeature"].values
y_compareProducts = df["compareProducts"].values
y_reviewFeature = df["reviewFeature"].values
y_imageGallery = df["imageGallery"].values
y_newsletterSubscription = df["newsletterSubscription"].values
y_postSubmission = df["postSubmission"].values
y_editFeature = df["editFeature"].values
y_deleteFeature = df["deleteFeature"].values
y_shareReferral = df["shareReferral"].values
y_paymentProcess = df["paymentProcess"].values
y_orderHistory = df["orderHistory"].values
y_shipmentTracking = df["shipmentTracking"].values
y_addressManagement = df["addressManagement"].values
y_calendarEvents = df["calendarEvents"].values
y_bookingFeature = df["bookingFeature"].values
y_forumDiscussion = df["forumDiscussion"].values
y_reportFeedback = df["reportFeedback"].values
y_ticketSubmission = df["ticketSubmission"].values
y_filterSort = df["filterSort"].values
y_categoryCollection = df["categoryCollection"].values
y_productListing = df["productListing"].values
y_invoiceFeature = df["invoiceFeature"].values

# Combine the labels into a dictionary
y = {
    "registerForm": y_registerForm,
    "rstPss": y_rstPss,
    "chatBot": y_chatBot,
    "bagCart": y_bagCart,
    "accountSettings": y_accountSettings,
    "couponPromotion": y_couponPromotion,
    "LoginLogout": y_LoginLogout,
    "contactForm": y_contactForm,
    "searchFeature": y_searchFeature,
    "linksPage": y_linksPage,
    "fileUpload": y_fileUpload,
    "fileDownload": y_fileDownload,
    "wishlistFeature": y_wishlistFeature,
    "compareProducts": y_compareProducts,
    "reviewFeature": y_reviewFeature,
    "imageGallery": y_imageGallery,
    "newsletterSubscription": y_newsletterSubscription,
    "postSubmission": y_postSubmission,
    "editFeature": y_editFeature,
    "deleteFeature": y_deleteFeature,
    "shareReferral": y_shareReferral,
    "paymentProcess": y_paymentProcess,
    "orderHistory": y_orderHistory,
    "shipmentTracking": y_shipmentTracking,
    "addressManagement": y_addressManagement,
    "calendarEvents": y_calendarEvents,
    "bookingFeature": y_bookingFeature,
    "forumDiscussion": y_forumDiscussion,
    "reportFeedback": y_reportFeedback,
    "ticketSubmission": y_ticketSubmission,
    "filterSort": y_filterSort,
    "categoryCollection": y_categoryCollection,
    "productListing": y_productListing,
    "invoiceFeature": y_invoiceFeature,
}

y_array = np.concatenate([df[key].values.reshape(-1, 1) for key in df.columns[1:]], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_array, test_size=0.2, random_state=42)

# Set the maximum sequence length and vocabulary size
max_sequence_length = 100
vocab_size = 10000
embedding_dim = 50

# Tokenize and pad the input sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, maxlen=max_sequence_length, padding="post")
X_test = pad_sequences(X_test, maxlen=max_sequence_length, padding="post")

# Define the model architecture
input_layer = layers.Input(shape=(max_sequence_length,))
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(input_layer)
lstm_layer = layers.LSTM(units=64, return_sequences=True)(embedding_layer)
dropout_layer = layers.Dropout(0.5)(lstm_layer)

outputs = []
for i in range(len(y_train[0])):
    output = layers.Dense(1, activation="sigmoid", name="output_{}".format(i + 1))(dropout_layer)
    outputs.append(output)

# Define the model
model = models.Model(inputs=input_layer, outputs=outputs)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Define the ModelCheckpoint callback
checkpoint_path = "model_checkpoint.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)

# Train the model with callbacks
model.fit(
    X_train,
    [y_train[:, i].reshape(-1, 1) for i in range(len(y_train[0]))],
    epochs=10,
    batch_size=32,
    validation_data=(
        X_test,
        [y_test[:, i].reshape(-1, 1) for i in range(len(y_test[0]))],
    ),
    callbacks=[early_stopping, model_checkpoint],
    verbose=2
)

# Evaluate the model
eval_results = model.evaluate(X_test, [y_test[:, i].reshape(-1, 1) for i in range(len(y_test[0]))])

# Unpack the evaluation results
loss = eval_results[0]
accuracies = eval_results[1:]

# Print the evaluation results
print("Test Loss: {:.4f}".format(loss))
for i, key in enumerate(y.keys()):
    accuracy = accuracies[i * 2]  # Only select the first accuracy for each category
    print("Accuracy {}: {:.2%}".format(key, accuracy))


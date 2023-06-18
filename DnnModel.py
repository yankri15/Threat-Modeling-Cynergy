import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

print(torch.cuda.is_available())


# Check CUDA availability
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")


# Load the preprocessed dataset
df = pd.read_csv("data.csv")

# Prepare the input data
X = df["sentence"].values

# Prepare the output data
output_columns = df.columns[1:]  # Exclude "sentence" column
y = df[output_columns].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the maximum sequence length and vocabulary size
max_sequence_length = 100
vocab_size = 10000
embedding_dim = 50

# Tokenize and pad the input sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequence([torch.tensor(seq) for seq in X_train], batch_first=True, padding_value=0)
X_test = pad_sequence([torch.tensor(seq) for seq in X_test], batch_first=True, padding_value=0)

# Create the PyTorch model
class ThreatModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ThreatModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return torch.sigmoid(output)

output_dim = len(output_columns)

model = ThreatModel(vocab_size, embedding_dim, 64, output_dim)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model and data to the device
model.to(device)
X_train = X_train.to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = X_test.to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define the batch size
batch_size = 32

# Create the training DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the testing DataLoader
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training
model.train()
for epoch in range(10):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")

# Evaluation
model.eval()
category_correct = [0] * output_dim
category_total = [0] * output_dim

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        predicted = torch.round(outputs)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            for j in range(output_dim):
                category_correct[j] += c[i, j].item()
                category_total[j] += 1

# Print accuracy for each category
for i, column in enumerate(output_columns):
    accuracy = category_correct[i] / category_total[i] * 100
    print(f"Accuracy {column}: {accuracy:.2f}%")

# #######################################################################################################################################################################
# OLD CODE
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras import layers, models

# # Load the preprocessed dataset
# df = pd.read_csv("data.csv")

# # Prepare the input data
# X = df["sentence"].values

# # Prepare the output data
# output_columns = df.columns[1:]
# y = {}
# for column in output_columns:
#     if column != "sentence":
#         y[column] = df[column].values.reshape(-1, 1)

# # Concatenate the output arrays
# y_array = np.concatenate([y[column] for column in output_columns if column != "sentence"], axis=1)


# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_array, test_size=0.2, random_state=42)

# # Set the maximum sequence length and vocabulary size
# max_sequence_length = 100
# vocab_size = 10000
# embedding_dim = 50

# # Tokenize and pad the input sequences
# tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)
# X_train = pad_sequences(X_train, maxlen=max_sequence_length, padding="post")
# X_test = pad_sequences(X_test, maxlen=max_sequence_length, padding="post")

# # Define the model architecture
# input_layer = layers.Input(shape=(max_sequence_length,))
# embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(input_layer)
# lstm_layer = layers.LSTM(units=64, return_sequences=True)(embedding_layer)
# dropout_layer = layers.Dropout(0.5)(lstm_layer)

# # each output layer will output a probability between 0 and 1, indicating the likelihood that a sentence contains
# outputs = []
# for i in range(len(y_train[0])):
#     output = layers.Dense(1, activation="sigmoid", name="output_{}".format(i + 1))(dropout_layer)
#     outputs.append(output)

# # Define the model
# model = models.Model(inputs=input_layer, outputs=outputs)

# # Compile the model
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Define the EarlyStopping callback
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# # Define the ModelCheckpoint callback
# checkpoint_path = "model_checkpoint.h5"
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)

# # Train the model with callbacks
# model.fit(
#     X_train,
#     [y_train[:, i].reshape(-1, 1) for i in range(len(y_train[0]))],
#     epochs=10,
#     batch_size=32,
#     validation_data=(
#         X_test,
#         [y_test[:, i].reshape(-1, 1) for i in range(len(y_test[0]))],
#     ),
#     callbacks=[early_stopping, model_checkpoint],
#     verbose=2
# )

# # Evaluate the model
# eval_results = model.evaluate(X_test, [y_test[:, i].reshape(-1, 1) for i in range(len(y_test[0]))])

# # Unpack the evaluation results
# loss = eval_results[0]
# accuracies = eval_results[1:]

# # Print the evaluation results
# print("Test Loss: {:.4f}".format(loss))
# for i, key in enumerate(y.keys()):
#     accuracy = accuracies[i * 2 + 1]  # Select the accuracy for each category
#     print("Accuracy {}: {:.2%}".format(key, accuracy))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.backends.cuda
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Check CUDA availability
if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device('cpu')

# If CUDA is available, use a collate_fn that moves data to the GPU
collate_fn = None
if device.type == 'cuda':
    def collate_fn(batch):
        return tuple(x_.to(device) for x_ in default_collate(batch))

# Load the preprocessed dataset
df = pd.read_csv("data.csv")

# Prepare the input data
X = df["sentence"].values

# Prepare the output data
output_columns = df.columns[1:]  # Exclude "sentence" column
y = df[output_columns].values

# Split the dataset into training, validation and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  

# Set the maximum sequence length and vocabulary size
max_sequence_length = 100
vocab_size = 10000
embedding_dim = 50

# Tokenize and pad the input sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequence([torch.tensor(seq) for seq in X_train], batch_first=True, padding_value=0)
X_val = pad_sequence([torch.tensor(seq) for seq in X_val], batch_first=True, padding_value=0)
X_test = pad_sequence([torch.tensor(seq) for seq in X_test], batch_first=True, padding_value=0)

# Create the PyTorch model
class ThreatModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5, num_layers=2):
        super(ThreatModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return self.sigmoid(x)

output_dim = len(output_columns)
model = ThreatModel(vocab_size, embedding_dim, 64, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Prepare the data for DataLoader
train_dataset = TensorDataset(X_train, torch.tensor(y_train).float())
val_dataset = TensorDataset(X_val, torch.tensor(y_val).float())
test_dataset = TensorDataset(X_test, torch.tensor(y_test).float())

n_epochs = 20
batch_size = 128

# Create the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Training loop
best_val_loss = float('inf')
patience = 5
counter = 0

train_accuracies = []
val_accuracies = []
train_losses_epoch = []

for epoch in range(n_epochs):
    model.train()
    train_losses = []
    train_correct = 0
    train_total = 0
    for i, (train_input, train_target) in enumerate(train_dataloader):
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        l1_lambda = 0.01
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = train_loss + l1_lambda * l1_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Calculate accuracy
        predicted = torch.round(train_output.data)
        correct_predictions = predicted.eq(train_target.data).view(-1)
        train_correct += torch.sum(correct_predictions).item()
        train_total += correct_predictions.shape[0]

    mean_train_loss = np.mean(train_losses)
    train_losses_epoch.append(mean_train_loss)
    
    train_accuracy = 100 * train_correct / train_total
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(train_losses)}")

    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for i, (val_input, val_target) in enumerate(val_dataloader):
            val_output = model(val_input)
            val_loss = criterion(val_output, val_target)
            val_losses.append(val_loss.item())
            
            # Calculate accuracy
            predicted = torch.round(val_output.data)
            correct_predictions = predicted.eq(val_target.data).view(-1)
            val_correct += torch.sum(correct_predictions).item()
            val_total += correct_predictions.shape[0]

    # val_accuracy = 100 * val_correct / val_total
    # val_accuracies.append(val_accuracy)
    mean_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {mean_val_loss}")

    scheduler.step(mean_val_loss)  # Update the learning rate scheduler

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping. No improvement in validation loss.")
            break

plt.figure(figsize=(10, 5))
plt.plot(train_losses_epoch, label='Training loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
test_losses = []
predicted_labels = []
with torch.no_grad():
    for i, (test_input, test_target) in enumerate(test_dataloader):
        test_output = model(test_input)
        test_loss = criterion(test_output, test_target)
        test_losses.append(test_loss.item())
        
        predicted_labels_batch = torch.round(test_output).cpu().numpy()
        predicted_labels.append(predicted_labels_batch)

mean_test_loss = np.mean(test_losses)
print(f"Test Loss: {mean_test_loss}")

# Concatenate the predicted labels from all batches
predicted_labels = np.concatenate(predicted_labels)

# Count number of predicted samples for each category
category_correct = [0] * len(output_columns)
category_total = [0] * len(output_columns)

model.eval()
with torch.no_grad():
    for i, (val_input, val_target) in enumerate(val_dataloader):
        val_output = model(val_input)
        predicted = torch.round(val_output.data)
        for j in range(len(output_columns)):
            category_correct[j] += (predicted[:, j] == val_target[:, j]).sum().item()
            category_total[j] += val_target.size(0)


for i, column in enumerate(output_columns):
    if category_total[i] != 0:
        accuracy = category_correct[i] / category_total[i] * 100
    else:
        accuracy = 0
    print(f"Accuracy {column}: {accuracy:.2f}%")


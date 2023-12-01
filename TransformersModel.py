import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

class ThreatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            "text": text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

df = pd.read_csv('data.csv')

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Split the data
train_data, test_data = train_test_split(df, test_size=0.2)
train_data, val_data = train_test_split(train_data, test_size=0.25)  # 0.25 x 0.8 = 0.2

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Create data loaders
train_dataset = ThreatDataset(
    texts=train_data["sentence"].to_list(),
    labels=train_data.drop("sentence", axis=1).values.tolist(),
    tokenizer=tokenizer,
    max_len=128
)

val_dataset = ThreatDataset(
    texts=val_data["sentence"].to_list(),
    labels=val_data.drop("sentence", axis=1).values.tolist(),
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = ThreatDataset(
    texts=test_data["sentence"].to_list(),
    labels=test_data.drop("sentence", axis=1).values.tolist(),
    tokenizer=tokenizer,
    max_len=128
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Initialize the model
num_labels = df.shape[1] - 1  # number of threat categories
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=num_labels
)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training setup
total_threats = 2268 + 2311 + 5936 + 5455 + 3040 + 2033
class_weights = [total_threats / threats for threats in [2268, 2311, 5936, 5455, 3040, 2033]]
class_weights = torch.FloatTensor(class_weights).to(device)
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)

best_f1 = 0.0  # track the best F1-score on validation data

for epoch in range(3):  # adjust the number of epochs as needed
    print(f'Epoch {epoch+1}')
    model.train()
    epoch_loss = 0.0
    epoch_preds = []
    epoch_labels = []
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = loss_fn(logits, labels)
        epoch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Add batch predictions and labels to the epoch lists
        preds = torch.sigmoid(outputs.logits) > 0.5
        epoch_preds.extend(preds.cpu().detach().numpy())
        epoch_labels.extend(labels.cpu().detach().numpy())
    
    print(f"Training Loss: {epoch_loss/len(train_loader)}")
        
    # Validation phase
    model.eval()
    epoch_val_preds = []
    epoch_val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Add batch predictions and labels to the epoch lists
            preds = torch.sigmoid(outputs.logits) > 0.5
            epoch_val_preds.extend(preds.cpu().detach().numpy())
            epoch_val_labels.extend(labels.cpu().detach().numpy())
            
    # After each epoch, compute validation metrics and print them
    val_report = classification_report(np.array(epoch_val_labels), np.array(epoch_val_preds))
    print(f'Validation Report: {val_report}')
    
    val_f1 = score(np.array(epoch_val_labels).flatten(), np.array(epoch_val_preds).flatten(), average='weighted')[2]
    
    if val_f1 > best_f1:
        torch.save(model.state_dict(), 'best_model.pt')
        best_f1 = val_f1

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# After training, evaluate the model on the test set
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Add batch predictions and labels to the epoch lists
        preds = torch.sigmoid(outputs.logits) > 0.5
        test_preds.extend(preds.cpu().detach().numpy())
        test_labels.extend(labels.cpu().detach().numpy())
        
# Compute test metrics
test_report = classification_report(np.array(test_labels), np.array(test_preds))
print(f'Test Report: {test_report}')
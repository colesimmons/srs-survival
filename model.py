#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Dependencies ---

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


# In[2]:


# --- Data preprocessing ---

# Load CSV
df = pd.read_csv('output.csv', parse_dates=['timestamp'])

# Sort dataframe by timestamp
df.sort_values(['card_id', 'timestamp'], ascending=True, inplace=True)

# Calculate time since last review for each review
df['time_since_last_review'] = df.groupby('card_id')['timestamp'].diff().dt.total_seconds().fillna(0) / (60 * 60 * 24)  # convert to days

# Calculate time since initial review for each review
df['time_since_initial_review'] = (df['timestamp'] - df.groupby('card_id')['timestamp'].transform('first')).dt.total_seconds() / (60 * 60 * 24)  # convert to days

# Drop cards with fewer than 4 reviews
df = df.groupby('card_id').filter(lambda x: len(x) >= 4)

# Normalize time columns
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df[['time_since_last_review', 'time_since_initial_review']])
df[['time_since_last_review', 'time_since_initial_review']] = normalized

# Drop 'timestamp'
df = df.drop(columns=['timestamp'])

unique_card_ids = df['card_id'].unique()
    
train_card_ids, test_card_ids = train_test_split(unique_card_ids, test_size=0.2, random_state=42)
df_train = df[df['card_id'].isin(train_card_ids)]
df_test = df[df['card_id'].isin(test_card_ids)]


# In[5]:


def fmt_sequences(df, balance=False):
    card_ids = df['card_id'].unique()

    sequences = [
        (
          df[df['card_id'] == card_id].iloc[:i].drop(columns=['card_id', 'was_remembered']).values,
          (df[df['card_id'] == card_id].iloc[i]['time_since_last_review'], df[df['card_id'] == card_id].iloc[i]['was_remembered'])
        ) for card_id in card_ids for i in range(2, len(df[df['card_id'] == card_id]))
    ]
    
    # shuffle the sequences
    random.shuffle(sequences)
    
    if balance is False:
        return sequences
    
    # separate positive and negative examples
    positive = [seq for seq in sequences if seq[1][1] == 1]
    negative = [seq for seq in sequences if seq[1][1] == 0]

    # oversample negative examples
    negative += random.choices(negative, k=len(positive) - len(negative))

    # combine positive and negative examples and shuffle them again
    balanced = positive + negative
    random.shuffle(balanced)
    return balanced


# In[6]:


train_sequences = fmt_sequences(df_train, balance=True)
test_sequences = fmt_sequences(df_test)


# In[7]:


X_train = [sequence for sequence, target in train_sequences]
X_test = [sequence for sequence, target in test_sequences]
Y_train = [target for sequence, target in train_sequences]
Y_test = [target for sequence, target in test_sequences]


# In[8]:



# --- Datasets ---
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).float()

# Pad the sequences and create your DataLoader
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pad sequences so they all have the same length within one batch
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    return inputs_padded, torch.stack(targets)

train_dataset = ReviewDataset(X_train, Y_train)
test_dataset = ReviewDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


# In[ ]:





# In[9]:


# --- Model ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define output layer
        self.fc = nn.Linear(hidden_size, 2)  # Two output nodes for λ and k

    def forward(self, x):
        # Set initial hidden states: (num_layers, batch_size, hidden_size)
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        y, hidden = self.lstm(x)  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(y[:, -1, :])  # only take the output from the last time step
        # Get parameters of the Weibull distribution
        λ, k = torch.exp(out[:, 0]), torch.exp(out[:, 1])  # Ensure positive values
        return λ, k


# In[10]:


def weibull_survival_function(λ, k, t):
    return torch.exp(- (t / λ) ** k)


# In[11]:


model = LSTMModel(input_size=3, hidden_size=64, num_layers=2)  # Adjust the parameters as needed
# model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed


# In[12]:


num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    model.train()

    for inputs, targets in train_loader:
        λ, k = model(inputs)
        review_time = targets[:, 0]
        remembered = targets[:, 1]
        
        # Compute predicted survival probabilities at the time of the next review
        predicted_survival = weibull_survival_function(λ, k, review_time)

        # Compute the binary cross-entropy loss between the predicted probabilities and the true outcomes
        loss = nn.BCELoss()(predicted_survival, remembered)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            λ, k = model(inputs)
            review_time = targets[:, 0]
            remembered = targets[:, 1]
            predicted_survival = weibull_survival_function(λ, k, review_time)
            preds = (predicted_survival > 0.5).float()
            all_preds.extend(preds)
            all_targets.extend(remembered)
    train_f1 = f1_score(all_targets, all_preds)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Train F1: {train_f1}')
    
    # Test
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            λ, k = model(inputs)
            review_time = targets[:, 0]
            remembered = targets[:, 1]
            predicted_survival = weibull_survival_function(λ, k, review_time)
            preds = (predicted_survival > 0.5).float()
            all_preds.extend(preds)
            all_targets.extend(remembered)
    test_f1 = f1_score(all_targets, all_preds)

    print(f'Test F1: {test_f1}')


# In[ ]:





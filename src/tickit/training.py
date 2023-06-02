from utils import preprocessing, generate_sequences, SequenceDataset
from network import LSTMForecaster

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path

pathdata = str(Path(__file__).parent/'dataset/sales_tab.txt')
pathout = str(Path(__file__).parent/'reports/')

# Here we are defining properties for our model
split = 0.8 # Train/Test Split ratio
sequence_len = 30 # training window (days)
nout = 7 # Prediction window (days)
BATCH_SIZE = 7 # Training batch size

# load and preprocess data
sales = pd.read_csv(pathdata, sep='\t', header = None, encoding= 'utf-8')
sales.columns =['salesid','listid','sellerid','buyerid','eventid','dateid','qtysold','pricepaid','commission','saletime']
df = sales[['qtysold', 'saletime']]
norm_df = preprocessing(df)

sequences = generate_sequences(norm_df, sequence_len, nout, 'qtysold')
dataset = SequenceDataset(sequences)

# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
train_ds, test_ds = data.random_split(dataset, lens)
# at each iteration the DataLoader will yield (batch size) sequences with their associated 
# targets which we will pass into the model
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

nhid = 50 # Number of nodes in the hidden layer
n_dnn_layers = 10 # Number of hidden fully connected layers

ninp = 1

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

# Initialize the model
model = LSTMForecaster(ninp, nhid, nout, sequence_len, device, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)

# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 50

# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Lists to store training and validation losses
t_losses, v_losses = [], []
worst_loss = 1000
# Loop over epochs
for epoch in range(n_epochs):
  train_loss, valid_loss = 0.0, 0.0

  # train step
  model.train()
  # Loop over train dataset
  for x, y in trainloader:
    optimizer.zero_grad()
    # move inputs to device
    x = x.to(device)
    y  = y.squeeze().to(device)
    # Forward Pass
    preds = model(x).squeeze()
    loss = criterion(preds, y) # compute batch loss
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  epoch_loss = train_loss / len(trainloader)
  t_losses.append(epoch_loss)
  
  # validation step
  model.eval()
  # Loop over validation dataset
  for x, y in testloader:
    with torch.no_grad():
      x, y = x.to(device), y.squeeze().to(device)
      preds = model(x).squeeze()
      error = criterion(preds, y)
    valid_loss += error.item()
  valid_loss = valid_loss / len(testloader)
  v_losses.append(valid_loss)
  
  if worst_loss > valid_loss:
    worst_loss = valid_loss

    state = {'epoch':epoch, 'state_dict':model.state_dict(),
             'optimizer': optimizer.state_dict(), 'tw':sequence_len}
    torch.save(state, pathout + '/best.pth')

  print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

## plotting
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(v_losses,label="val")
plt.plot(t_losses,label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(pathout + '/training.png')

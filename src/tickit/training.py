from utils import preprocessing, generate_sequences, SequenceDataset
from network import LSTMForecaster

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn as nn

# Here we are defining properties for our model
BATCH_SIZE = 16 # Training batch size
split = 0.8 # Train/Test Split ratio
steps_to_predict = 7

pathdata = "./dataset/sales_tab.txt"
norm_df = preprocessing(pathdata)

sequences = generate_sequences(norm_df.qtysold.to_frame(), len(norm_df), steps_to_predict, 'qtysold')
dataset = SequenceDataset(sequences)

# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
train_ds, test_ds = data.random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

nhid = 50 # Number of nodes in the hidden layer
n_dnn_layers = 5 # Number of hidden fully connected layers
nout = steps_to_predict # Prediction Window
sequence_len = len(norm_df) # Training Window

# Number of features (since this is a univariate timeseries we'll set
ninp = 1

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

# Initialize the model
model = LSTMForecaster(ninp, nhid, nout, sequence_len, device, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)

# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 20

# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Lists to store training and validation losses
t_losses, v_losses = [], []
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
      
  print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

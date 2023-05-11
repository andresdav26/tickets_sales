from utils import preprocessing, generate_sequences, preprocessing_inference, SequenceDataset
from network import LSTMForecaster

import torch
from torch.utils.data import DataLoader

import pandas as pd
from pathlib import Path

def make_predictions(model, dataloader):
  model.eval()
  predictions = []
  for x, y in dataloader:
    with torch.no_grad():
      p = model(x)
      predictions.append(p)
  predictions = torch.cat(predictions).numpy()
  return predictions.squeeze()

pathdata = str(Path(__file__).parent/'dataset/sales_tab.txt')
model_path = str(Path(__file__).parent.parent/'tickit/reports/best.pth')

# load and preprocess data
sales = pd.read_csv(pathdata, sep='\t', header = None, encoding= 'utf-8')
sales.columns =['salesid','listid','sellerid','buyerid','eventid','dateid','qtysold','pricepaid','commission','saletime']
df = sales[['qtysold', 'saletime']]
data = preprocessing(df)

answer = {'prediction':{}}

# load model
# the model requires "sequence_len" (training window) samples to run.
# and predice 7 samples. 
checkpoint = torch.load(model_path)
sequence_len = checkpoint['tw']
model = LSTMForecaster(1, 50, 7, sequence_len, 'cpu', n_deep_layers=5)
model.load_state_dict(checkpoint['state_dict'])


# at least "sequence_len"" samples are required.
data.drop(data.index[:-sequence_len], inplace=True)
sequences = generate_sequences(data, len(data), 7, 'qtysold')
dataset = SequenceDataset(sequences)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

prediction = make_predictions(model, dataloader)
for ans in range(len(prediction)):
    answer['prediction'][f'day {ans+1}'] = str(prediction[ans])
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class SequenceDataset(torch.utils.data.Dataset):

  def __init__(self, df):
    self.data = df

  def __getitem__(self, idx):
    sample = self.data[idx]
    return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])
  
  def __len__(self):
    return len(self.data)
  

def preprocessing(data:pd.DataFrame) -> pd.DataFrame:

    data['saletime'] = pd.to_datetime(data['saletime'])
    data['saletime'] = data['saletime'].dt.strftime('%Y-%m-%d')
    data['saletime'] = data['saletime'].astype('datetime64[ns]')
    data = data.set_index('saletime')
    data = data.sort_index()
    daily = data.resample('D').sum()
    
    # Normalize
    # Fit scalers
    scalers = {}
    for x in daily.columns:
        scalers[x] = StandardScaler().fit(daily[x].values.reshape(-1, 1))

    # Transform data via scalers
    norm_df = daily.copy()
    for i, key in enumerate(scalers.keys()):
        norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
        norm_df.iloc[:, i] = norm

    return norm_df

def preprocessing_inference(data:pd.DataFrame) -> pd.DataFrame:

    data['saletime'] = pd.to_datetime(data['saletime'])
    data['saletime'] = data['saletime'].dt.strftime('%Y-%m-%d')
    data['saletime'] = data['saletime'].astype('datetime64[ns]')
    data = data.set_index('saletime')
    data = data.sort_index()
    daily = data.resample('D').sum()
    
    return daily

# Defining a function that creates sequences and targets as shown above
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns):
  '''
  df: Pandas DataFrame of the univariate time-series
  tw: Training Window - Integer defining how many steps to look back
  pw: Prediction Window - Integer defining how many steps forward to predict

  returns: dictionary of sequences and targets for all sequences
  '''
  data = dict() # Store results into a dictionary
  L = len(df)

  if L == tw: 
     data[0] = {'sequence': df.values, 'target': np.array([])}

  else:
    for i in range(L-tw-pw+1):
      # Get current sequence  
      sequence = df[i:i+tw].values
      # Get values right after the current sequence
      target = df[i+tw:i+tw+pw][target_columns].values
      data[i] = {'sequence': sequence, 'target': target}

  return data
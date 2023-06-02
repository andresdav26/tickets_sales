from src.tickit.utils import preprocessing_inference, generate_sequences, SequenceDataset
from src.tickit.network import LSTMForecaster

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pathlib import Path
import pandas as pd
from typing import List

from torch.utils.data import DataLoader
import torch


model_path = str(Path(__file__).parent.parent/'tickit/reports/best.pth')


def make_predictions(model, dataloader):
  model.eval()
  predictions = []
  for x, y in dataloader:
    with torch.no_grad():
      p = model(x)
      predictions.append(p)
  predictions = torch.cat(predictions).numpy()
  return predictions.squeeze()


app = FastAPI(title='API SALES FORECAST')

class Historial(BaseModel):
    qtysold: List[int] = []
    saletime: List[str] = []
    

@app.get('/')
def root():
    html_content = """
    <html>
        <meta http-equiv=”Content-Type” content=”text/html; charset=UTF-8″ />
        <head>
            <title>API SALES FORECAST</title>
        </head>
        <body>
            <h1>Sales forecast of tickets</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict/")
async def predict(input: Historial):

    if input is not None:
        answer = {'prediction':{}}
        
        # load model
        # the model requires "sequence_len" (training window) samples to run.
        # and predice 7 samples. 
        checkpoint = torch.load(model_path)
        sequence_len = checkpoint['tw']
        model = LSTMForecaster(1, 50, 7, sequence_len, 'cpu', n_deep_layers=10)
        model.load_state_dict(checkpoint['state_dict'])

        # get data
        data = input.dict()
        data = pd.DataFrame.from_dict(data)
        data = preprocessing_inference(data)

        if len(data) < sequence_len: 
           return {'WarningError': 'At least "sequence_len" samples are required'}
        
        # at least "sequence_len"" samples are required.
        data.drop(data.index[:-sequence_len], inplace=True)
        sequences = generate_sequences(data, len(data), 7, 'qtysold')
        dataset = SequenceDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        prediction = make_predictions(model, dataloader)
        for ans in range(len(prediction)):
            answer['prediction'][f'day {ans+1}'] = str(prediction[ans])

        return answer

    else:
        return {'WarningError': 'Model not found'}


if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port=8000)

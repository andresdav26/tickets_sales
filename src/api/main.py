import logging
import uvicorn
import pandas as pd
import numpy as np
import os
import yaml

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from utils import preprocessing

from typing import List, Optional, Union, Tuple, Any


app = FastAPI(title='API SALES FORECAST')

class Historial(BaseModel):
    saletime: List[str] = []
    qtysold: List[int] = []

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

    pass


if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port=8000)

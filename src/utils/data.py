import pandas as pd 

def preprocessing(pathdata:str) -> pd.DataFrame:

    sales = pd.read_csv(pathdata, sep='\t', header = None, encoding= 'utf-8')
    sales.columns =['salesid','listid','sellerid','buyerid','eventid','dateid','qtysold','pricepaid','commission','saletime']
    data = sales[['qtysold', 'saletime']]

    data['saletime'] = pd.to_datetime(data['saletime'])
    data['saletime'] = data['saletime'].dt.strftime('%Y-%m-%d')
    data['saletime'] = data['saletime'].astype('datetime64[ns]')
    data = data.set_index('saletime')
    data = data.sort_index()
    daily = data.resample('D').sum()

    return daily


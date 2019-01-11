from django import forms
import pandas as pd
import requests

data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
table = data[0]
sliced_table = table[1:]
header = table.iloc[0]
corrected_table = sliced_table.rename(columns=header)
tickers = corrected_table['Symbol'].tolist()

ASSET_CHOICES = (
    [(f"{tickers}",f"{tickers}") for tickers in tickers]
    )

data={}
def coin_list():
    url = 'https://www.cryptocompare.com/api/data/coinlist/'
    page = requests.get(url)
    data = page.json()['Data']
    return data
data = coin_list()
symbols=list()
for name in data.keys():
    symbols = symbols+[data[name]["Symbol"]]
    

CRYPTO_CHOICES =(
    [(f"{cryptos}",f"{cryptos}") for cryptos in symbols]
    )

class HomeForm(forms.Form):
    None
    
class SP500Form(forms.Form):
    asset = forms.CharField(widget=forms.Select(choices=ASSET_CHOICES), max_length=1)

class CryptoForm(forms.Form):
    asset = forms.CharField(widget=forms.Select(choices=CRYPTO_CHOICES), max_length=1)
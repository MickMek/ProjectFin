
# coding: utf-8

# ## BTC Best Bid/Ask Analysis - Creating and analysis dataframe (PostgreSQL)

# In[1]:


# Go to the terminal and create table
# psql -U mickael cryppro_v0


# In[2]:


import slumber
from datetime import datetime
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from decimal import Decimal as D
import logging
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
import copy
import numpy as np
import time
import os
from sqlalchemy import create_engine
import psycopg2
import sys

class Exchange:
  def __init__(self, url, commissions):
    self.url = url
    self.commissions = commissions
    self.api = None
    self.session = None

  def get_requests_session(self):
    if self.session is not None:
      return self.session

    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 404, 502, 503, 504 ])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    self.session = s
    return s

  def get_slumber_api(self, append_slash=True):
    if self.api is not None:
      return self.api

    s = self.get_requests_session()
    self.api = slumber.API(self.url, session=s, append_slash=append_slash)
    return self.api

  def fetch(self, ticker):
    raise ValueError("Not implemented")

  def postprocess(self, ob1, ticker):
    raise ValueError("Not implemented")



class Kraken(Exchange):
  def __init__(self):
    super().__init__(
      url="https://api.kraken.com/0/public/",
      commissions = D('0.14')/100
    )

  def fetch(self, ticker):
    logging.debug('fetch kraken')

    api = self.get_slumber_api(append_slash=False)
    ob1 = api.Depth.get(pair=ticker, count=30)
    return ob1

  def postprocess(self, ob1, ticker):
    """
    ob1 - output of self.fetch(ticker)
    """
    logging.debug('postprocess kraken')
    
    if ob1['error']:
        raise ValueError("Kraken error: %s"%(ob1['error']))

    ob2 = copy.deepcopy(ob1)
    ob2 = list(ob2['result'].values())[0]
    for k1 in ['asks', 'bids']:
        ob2[k1] = pd.DataFrame(ob2[k1]).rename(columns={0:'price', 1:'amount', 2:'timestamp'})
        # convert to Decimal
        for k2 in ['price', 'amount']:
            ob2[k1][k2] = [D(x) for x in ob2[k1][k2].values]

    # commissions
    ob2['asks']['price pre-commission'] =  ob2['asks']['price']
    ob2['bids']['price pre-commission'] =  ob2['bids']['price']
    ob2['asks']['price'] = ob2['asks']['price'] * (1+self.commissions)
    ob2['bids']['price'] = ob2['bids']['price'] * (1-self.commissions)

    # set side
    ob2['asks']['side'] = 'ask'
    ob2['bids']['side'] = 'bid'

    ob3 = pd.concat([ob2['asks'], ob2['bids']], axis=0).reset_index(drop=True)
    ob3['source'] = 'kraken.com'
    ob3['ticker'] = ticker

    return ob3


kraken = Kraken()
ticker = 'XBTUSD'
order_book = kraken.fetch(ticker)
order_book = kraken.postprocess(order_book, ticker)
        
#Obtaining min_ask and max_bid for current timestamp
min_ask = order_book[order_book['side']=='ask']['price'].min()
max_bid = order_book[order_book['side']=='bid']['price'].max()

fn = '/home/mickael/1-btc_order_book_monitor/db_pwd.secret'
pwd = open(fn, 'r').readlines()[0]
pwd = pwd.strip()
engine = create_engine("postgresql+psycopg2://mickael:%s@localhost/cryppro_v0"%pwd)
df3 = pd.read_sql_table('btcprice', con=engine)
data = pd.DataFrame({"minimum_ask": float(min_ask), "maximum_bid": float(max_bid), "timestamp": str(datetime.now()), "price": float((min_ask+max_bid)/2), "spread": float(min_ask-max_bid)}, index=['%i'%len(df3)])

data.to_sql('btcprice', con=engine, if_exists='append')


# In[3]:


data


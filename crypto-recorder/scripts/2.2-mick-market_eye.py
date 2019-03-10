
# coding: utf-8

# ## Market Eye - Alert System for daily >5% price crash

# In[43]:


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
import smtplib
from sqlalchemy import create_engine
import psycopg2
import sys

fn = '/home/mickael/1-btc_order_book_monitor/db_pwd.secret'
pwd = open(fn, 'r').readlines()[0]
pwd = pwd.strip()
engine = create_engine("postgresql+psycopg2://mickael:%s@localhost/cryppro_v0"%pwd)

df2 = pd.read_sql_table('btcprice', con=engine)
d = datetime.now()
only_date, only_time = str(d.date()), str(d.time())

df = pd.DataFrame({})
for a in range(0,len(df2)):
    if only_date not in df2.iloc[a,1]:
        ''
    else:
        df = df.append(df2.iloc[a])
today_open, last_price = df.iloc[0,3], df.iloc[len(df)-1,3]
ROC_price = (last_price - today_open)/today_open

if ROC_price < (-0.05):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', port=587)
    server.connect('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    fn = '/home/mickael/1-btc_order_book_monitor/db_pwd.secret'
    pwd = open(fn, 'r').readlines()[1]
    pwd = pwd.strip()
    server.login("market.eye.alerts@gmail.com", pwd)
    msg = "ALERT! CRASH >5%"
    server.sendmail("market.eye.alerts@gmail.com", ["market.eye.alerts@gmail.com","mickael.mekari@gmail.com","shadi@akikieng.com"], msg) 
    server.quit()
if ROC_price > (0.05):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', port=587)
    server.connect('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    fn = '/home/mickael/1-btc_order_book_monitor/db_pwd.secret'
    pwd = open(fn, 'r').readlines()[1]
    pwd = pwd.strip()
    server.login("market.eye.alerts@gmail.com", pwd)
    msg = "ALERT! PRICE HIKE >5%"
    server.sendmail("market.eye.alerts@gmail.com", ["market.eye.alerts@gmail.com","mickael.mekari@gmail.com","shadi@akikieng.com"], msg) 
    server.quit()
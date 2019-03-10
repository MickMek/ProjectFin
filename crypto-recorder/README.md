# BITCOIN(BTC) Order Book Monitor

This folder is composed of 4 applications, each as notebooks (*.ipynb) and scripts (*.py), found in their respective folders. They are described in detail below.
crontabs jobs have been set up to run the applications scripts every hour. 

## Application I - "1.2-mick-recording_best_bid_ask"
Recording metrics (best bid/ask, spread, price) related to Bitcoin from Kraken.com.
The data is stored in the cryppro_v0 database as "btcprice" table (SELECT * FROM btcprice; to see recorded metrics).
This application will allow to store and build up a historical database of Bitcoin's spread.

## Application II - "1.3-mick-recording_lobs"
Recording Bitcoin's complete Limit Order Books from Kraken.com
The application download's current information on Bitcoin's orderbooks, such as bid/ask prices and volumes, and appends the new data with a new timestamp to previous orderbooks.
The data is stored in the cryppro_v0 database as "btclobs" table (SELECT * FROM btclobs; to see recorded order books).
This application will allow to store and build up a historical database of Bitcoin's order books.

## Application III - "2.2-mick-market_eye"
Monitoring Bitcoin's price movement by comparing each hour's price to the day's open price. If a decrease in the current's price to open price is superior to 5%, an email is sent to interested parties.

## Application IV - "3.0-mick-lob_analysis" (ONLY Jupyter Notebook `.ipynb` available)
Analysing recorded orderbooks (obtained from Application III) for market-making profitability. The application finds each recorded hour's best bid and ask prices
(`df2` best hourly bid/ask)
The application also simulates a market-maker that continuously aggregate bids and asks from the executed trades until one side reaches 10 BTC. When this limit is reached, the market maker stops accumulating that side from the executed trades and only accumulate trades from the other side for the rest of the recorded hour.
(`df9` market maker inventory)

## Plotting

Upon receipt of a market crash/hike alert, notebooks `2.1` plot the database tables

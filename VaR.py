'''
Objective : create functions to calculate portfolio 1 day Value at Risk 

funciton1 : HistoricalSimulationVaR (Inputs : historical stock returns of the portfolio, CI, weights for each stock)

function2 : ParametricVaR (Inputs : Distribution for returns, historical stock returns, correlation_matrix, CI)
'''

# import libraries
import pandas as pd
import yfinance as yf
import random
import numpy as np
import math

# get price/ returns data from yfinance
def historicalreturns(ticker_list, start_date = '2024-01-01', end_date = '2025-01-01') :
    '''
    returns a dictionary with daily log returns for each ticker in the ticker_list from yfinance
    
    Input:
    ticker_list : list
    list of tickers
    start_date : str
    start date to fetch price. Format = 'YYYY-MM-DD'. default value = '2024-01-01'
    end_date : str
    end date to fetch price. Format = 'YYYY-MM-DD'. default value = '2025-01-01'

    Output:
    daily_returns : dictionary
    dictionary with ticker as key and returns as value
    
    '''

    # dictionary of tickers and daily returns
    daily_returns = {}

    # convert to list if only single ticker is passed in argument tickers
    if isinstance(ticker_list, str) :
        ticker_list = [ticker_list]

    for ticker in ticker_list :

        df = yf.download(ticker, start=start_date, end=end_date, progress=False).stack(future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)       # download data from yfinance
        df.columns.name = None # drop column name - 
        df['Returns'] = np.log(df['Close']).diff()  # add log returns
        df = df.dropna(subset=['Returns'])
        daily_returns[ticker] = df['Returns'] # save to dictionary daily_returns 
    
    daily_retuns_df = pd.DataFrame(daily_returns)  # convert to dataframe

    return daily_retuns_df

# function1 : Historical Simulation VaR
def historicalsimulationVaR(returns_df, stock_weights, portfolio_value, CI) :
    '''
    Calculates the VaR (value at risk) using historical simulation
    
    Input:
    returns_df : pd.Dataframe
    Daily returns data
    stock_weights : np.array
    weight of the stock
    portfolio_value : float
    Current value of the porfolio
    CI : float
    percentile to calculate VaR. Range (0%, 100%)

    Output
    VaR, VaR_Date, Individual_stock_returns : tuple
    Function return the 1-day VaR at CI, the date in the historical period,  returns of the individual stock
    ''' 

    # check - stock weights must sum to 1.0
    if not np.isclose(stock_weights.sum(), 1.0) :
        raise ValueError(f"Stock_weights array must sum to 1. Got {stock_weights.sum()} instead.")

    returns_df['PortfolioReturn'] = returns_df.to_numpy() @ stock_weights  # dot product of i.) daily returns and ii.) portfolio weights

    # sort by PortfolioReturns low to high
    returns_df_sorted = returns_df.sort_values(by='PortfolioReturn')

    # percentile index
    perc_index = math.ceil(len(returns_df_sorted) * (1 - CI))

    # VaR
    VaR = returns_df_sorted.iloc[perc_index - 1]['PortfolioReturn'] * portfolio_value

    # VaR Date
    VaR_Date = returns_df_sorted.iloc[perc_index - 1].name.strftime('%Y-%m-%d')

    return VaR, VaR_Date, returns_df_sorted.iloc[perc_index - 1]    # return a tuple



if __name__ == "__main__" :

    # fetch s&p500 companies from wiki
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"      # url to fetch data
    sp500_df = pd.read_html(wiki_url)[0]        # read from url
    sp500_df = sp500_df[~sp500_df['Symbol'].isin(['BRK.B', 'BF.B'])]  # drop some companies due to ticker mismatch
    sp500_df[['Symbol', 'Security']].to_csv('sp500.csv', index= False)  # save company list to folder

    # select stocks for portfolio
    alltickerlist = pd.read_csv('sp500.csv')['Symbol'].to_list()  # all ticker list
    portfolio1 = list(random.sample(alltickerlist, 5))  # portfolio1 stock tickers list : random select 5 stocks from all ticker list
    
    # get returns data from yfinance
    portfolio_individual_stock_returns_df = historicalreturns(portfolio1)

    # historical var portfolio1
    var_historicalsim = historicalsimulationVaR(portfolio_individual_stock_returns_df, np.array([0.1,0.3,0.7,0.05,-0.15]), 100, 0.95) 

    print(var_historicalsim[0])
    





    












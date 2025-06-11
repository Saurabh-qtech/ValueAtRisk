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
import scipy.stats as stats

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

    returns_df = returns_df.copy()

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

# function2 : parametric VaR
def parametricVaR (returns_df, stock_weights, portfolio_value, CI, return_dist_assumption, distribution_fit_params, **dist_params) :
    '''
    Calculates the VaR (value at risk) using parametric method. 
    This function works well with commmonly assumed districutions for 1-day asset returns 
    
    Input:
    returns_df : pd.Dataframe
    Daily returns data
    
    stock_weights : np.array
    weight of the stock
    
    portfolio_value : float
    Current value of the porfolio
    
    CI : float
    percentile to calculate VaR. Range (0%, 100%)
    
    return_dist_assumption : scipy.stats object
    distribution followed by daily return

    distribution_fit_params : bool
    1 : allow distribution .fit method to determine all necessary parameter including mean and std dev
    0 : mean and std dev are based on portfolio returns rather than .fit method

    Output
    VaR,  : tuple
    Function return the 1-day VaR at CI, parameters of the portfolio return
    
    '''
    # check - stock weights must sum to 1.0
    if not np.isclose(stock_weights.sum(), 1.0) :
        raise ValueError(f"Stock_weights array must sum to 1. Got {stock_weights.sum()} instead.")


    # calculate portfolio returns
    portfolio_returns = returns_df.to_numpy() @ stock_weights  # portfolio returns (@ is the matrix multiplication operator)

    
    port_mean = portfolio_returns.mean()  # mean of the portfolio
    port_var = portfolio_returns.var()   # variance of the portfolio
    port_stddev = np.sqrt(port_var)  # std dev of the portfolio
    '''
    params = {**dist_params, 'loc': port_mean, 'scale': port_stddev}   # parameters of the distribution
    '''
    # fit data to distribution
    if distribution_fit_params == 1 :
        params = return_dist_assumption.fit(portfolio_returns,  **dist_params)
    else:
        try :
            params = return_dist_assumption.fit(portfolio_returns, floc=port_mean, fscale=port_stddev, **dist_params )
        except ValueError as e :
            params = return_dist_assumption.fit(portfolio_returns, **dist_params)
    

    # calculate (1 - CI) in distribution
    VaR = return_dist_assumption.ppf(1-CI, *params) * portfolio_value

    # return parametric VaR
    return VaR, params




if __name__ == "__main__" :

    # fetch s&p500 companies from wiki
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"      # url to fetch data
    sp500_df = pd.read_html(wiki_url)[0]        # read from url
    sp500_df = sp500_df[~sp500_df['Symbol'].isin(['BRK.B', 'BF.B'])]  # drop some companies due to ticker mismatch
    sp500_df[['Symbol', 'Security']].to_csv('sp500.csv', index= False)  # save company list to folder

    # select stocks for portfolio
    alltickerlist = pd.read_csv('sp500.csv')['Symbol'].to_list()  # all ticker list
    portfolio1 = list(random.sample(alltickerlist, 5))  # portfolio1 stock tickers list : random select 5 stocks from all ticker list
    print(portfolio1)
    
    # get returns data from yfinance
    portfolio_individual_stock_returns_df = historicalreturns(portfolio1)

    # var portfolio1
    var_historicalsim = historicalsimulationVaR(portfolio_individual_stock_returns_df, np.array([0.1,0.3,0.7,0.05,-0.15]), 100, 0.95)   # historical var on portfolio1
    var_parametric_t = parametricVaR(portfolio_individual_stock_returns_df, np.array([0.1,0.3,0.7,0.05,-0.15]), 100, 0.95, stats.t,0)       # parametric var on portfolio1
    var_parametric_norm = parametricVaR(portfolio_individual_stock_returns_df, np.array([0.1,0.3,0.7,0.05,-0.15]), 100, 0.95, stats.norm,0)       # parametric var on portfolio1
    var_parametric_t_df10 = parametricVaR(portfolio_individual_stock_returns_df, np.array([0.1,0.3,0.7,0.05,-0.15]), 100, 0.95, stats.t,0, f0 = 10)       # parametric var on portfolio1
    # print historical var
    print('VaR tells that with there is a 5 percent chance that the loss exceeds the some value X')
    print('\n1-day 95% VaR')
    print(f'Historical Simulation : $ {-var_historicalsim[0]:.3f}')
    print(f'Parametric t-distribution : $ {-var_parametric_t[0]:.3f}')
    print(f'Parametric normal-distribution : $ {-var_parametric_norm[0]:.3f}')
    print(f'Parametric t-distribution with df = 10 : $ {-var_parametric_t_df10[0]:.3f}')
    print('What does a lower VaR mean for same level of confidence? It means that one method is projecting higher potential losses at the same confidence')






    












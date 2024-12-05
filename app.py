# Import base libraries
from numpy import *
import numpy as np
from numpy.linalg import multi_dot
import pandas as pd
import yfinance as yf
# Ignore warnings
# import warnings
# warnings.filterwarnings('ignore')
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
# !pip install -q streamlit
import streamlit as st
import scipy.optimize as sco

# ''' etfs
# * GLD: gold
# * EMB: Emerging markets bonds etf
# * AGG: USA bonds etf
# * QQQ: Nasdaq etf
# * SPEM: Emerging markets equity etf
# * MXN=X: USD/MXN'''

# Porfolio tickers.
tickers = ["EMB","AGG","QQQ","GLD","SPEM"]
tickers = list(sort(tickers))

# number of assets in portfolio
n = len(tickers)

# function to download data from yahoofinance
def download_data(assets, start_date, end_date):
  data = yf.download(assets, start = start_date, end = end_date)["Close"]
  return data

# begining and and dates for analysis
inicio = "2010-01-01"
fin = "2023-12-31"

# download assets and USD/MXN info
df = download_data(tickers, inicio, fin)
mxn = download_data("MXN=X", inicio, fin)

# calculating assets prices in mexican pesos
mxn = df.join(mxn)
df = mxn.drop(columns = ['MXN=X'])
mxn = mxn['MXN=X']
df_mxn = df.mul(mxn, axis = 0)

# Daily returns:
returns = df_mxn.pct_change().dropna()
returns_test = returns.loc["2010-01-01":"2020-12-31"]
# Summary statistics
means = returns_test.mean()
sds = returns_test.std()
skews = returns_test.skew()
kurtosis_excess = returns_test.kurtosis()
VaRs = returns_test.quantile(0.05)

# Expected shortfall function
def calcular_cvar(x, alpha):
  VaR = np.quantile(x,1-alpha)
  cVaR = x[x.lt(VaR)].mean()
  return(cVaR)


cVaRs = returns_test.apply(calcular_cvar,args = (0.95,),axis=0)


# risk free rate: 1 year treasuries. 4.297, updates december 2nd 2024.
rf = 0.04297

# sharpe ratio function
def sharpe_ratio(x, rf):
  dif = x-rf
  return(dif.mean()/dif.std())

sharpes = returns_test.apply(sharpe_ratio, args = (rf/252,), axis = 0)

# sortino ratio function
def sortino_ratio(x, rf):
  dif = x-rf
  return(dif.mean()/dif[dif<0].std())

sortinos = returns_test.apply(sortino_ratio, args = (rf/252,), axis = 0)

# max drawdon 
def drawdon(x):
  cum_returns = (1+x).cumprod()
  max_cum_return = cum_returns.cummax()
  drawdown = (max_cum_return - cum_returns)/max_cum_return
  max_drawdown = drawdown.max()
  return(max_drawdown)

drawdowns = returns_test.apply(drawdon, axis=0)

# joining all summary statistics in a dataframe
summary_df = pd.DataFrame([means,sds,skews,kurtosis_excess,VaRs,cVaRs,sharpes,sortinos,drawdowns],
                          index = ['mean','sd','skew','kurtosis','VaR 95%','cVaR 95%', 'sharpe ratio','sortino ratio','max drawdon'])

# Markowitz

# initial weights for every asset: equal weights.
initial_wts = (np.ones(n)/n)[:,newaxis]

# annualized daiy returns.
ret_an = array(returns.mean()*252)[:,newaxis]

# a function that will calculate annualized returns, volatility and sharpe ratio for the portfolio

def portfolio_stats(weights):

    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns_test.mean() * 252)[:,newaxis]
    port_vols = sqrt(multi_dot([weights.T, returns_test.cov() * 252, weights]))

    return array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Max Sharpe Ratio Portfolio
# Maximizing sharpe ratio by minimizing the negative sharpe ratio
def neg_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]

# Specify constraints and bounds
cons_max_sharpe = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
bnds_max_sharpe = tuple((0, 1) for x in range(n))
initial_wts_max_sharpe = np.ones(n)/n
# Optimizing portfolio
max_sharpe_port = sco.minimize(neg_sharpe_ratio, initial_wts_max_sharpe, method = 'SLSQP', bounds = bnds_max_sharpe, constraints = cons_max_sharpe)
# Portfolio weights
max_sharpe_port_wts = list(zip(tickers, around(max_sharpe_port['x']*100,2)))
# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
max_sharpe_port_stats = list(zip(stats, around(portfolio_stats(max_sharpe_port['x']),4)))

# Minimum Volatility Portfolio
def min_volatility(weights):
    return portfolio_stats(weights)[1]

# Specify constraints and bounds
cons_min_vol = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
bnds_min_vol = tuple((0, 1) for x in range(n))
initial_wts_min_vol = np.ones(n)/n
# Optimizing portfolio
min_vol_port = sco.minimize(min_volatility, initial_wts_min_vol, method = 'SLSQP', 
                            bounds = bnds_min_vol, constraints = cons_min_vol)
# Portfolio weights
min_vol_port_wts = list(zip(tickers, around(min_vol_port['x']*100,2)))
# Portfolio stats
min_vol_port_stats = list(zip(stats, around(portfolio_stats(min_vol_port['x']),4)))

# Efficient Frontier
# Minimize the volatility
def min_volatility(weights):
    return portfolio_stats(weights)[1]

# Efficient frontier params
targetrets = linspace(0.1,0.60,100)
tvols = []
weights_list = []

for tr in targetrets:

    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: sum(x) - 1})

    opt_ef = sco.minimize(min_volatility, initial_wts_min_vol, method='SLSQP', bounds=bnds_min_vol, constraints=ef_cons)

    tvols.append(opt_ef['fun'])
    weights_list.append(opt_ef['x'])

targetvols = array(tvols)
# Dataframe for EF
efport = pd.DataFrame({
    'targetrets' : around(100*targetrets,2),
    'targetvols': around(100*targetvols,2),
    'targetsharpe': around(targetrets/targetvols,2),
    'weights': weights_list
})

# Extracting weights and stats for the 10% return portfolio
ret_10_port_wts = list(zip(tickers, around(100*efport.iloc[0,3],2)))
ret_10_port_stats = list(zip(stats, around(portfolio_stats(efport.iloc[0,3]),4)))

# Backtesting
# S&P500 en pesos
spx = download_data("^GSPC", "2021-01-01", fin)

# calculating assets prices in mexican pesos
spx = spx.join(df)
spx = spx['^GSPC']
spx_mxn = spx.mul(mxn, axis = 0)["2021-1-01":"2023-12-31"]

# All portfolios and benchmark dataframe
backtest_prices = df_mxn["2021-01-01":"2023-12-31"]

# Black-Litterman
# Prior Distribution
# equally-weighted portfolio
prior_wts = np.ones(n)/n
# 0.5 sharpe-ratio
bl_sharpe = 0.5
# assets returns' covariance matrix
returns_cov = returns_test.cov()
# Prior portfolio volatility
prior_vol = sqrt(prior_wts.T@returns_cov@prior_wts)
# Risk aversion factor = sharpe*(1/vol)
risk_aversion = bl_sharpe/prior_vol
# prior distribution mean (Normal distribution).
prior_mean = array(risk_aversion*returns_cov@prior_wts)[:,newaxis]
# tau factor = 1/number of observations
tau = 1/len(returns_test)
# prior distribution covariance matrix
prior_cov = tau*returns_cov

# Investor Views and Posterior Weights
P = array([[1,0,0,0,0],
           [0,1,0,0,0],
           [0,0,1,0,0],
           [0,0,0,1,0],
           [0,0,0,0,1]])

Q = array([[0.1],[-0.08],[0.05],[0.2],[0.1]])

omega = np.diag(diag(P@prior_cov@P.T))

# Posterior distribution and weights
posterior_mean = np.linalg.inv(np.linalg.inv(prior_cov)+P.T@np.linalg.inv(omega)@P)@(np.linalg.inv(prior_cov)@prior_mean+P.T@np.linalg.inv(omega)@Q)
bl_port_wts = ((1/risk_aversion)*np.linalg.inv(returns_cov))@posterior_mean


# 2010-2020 portfolio summary
max_sharpe_port_data = np.concatenate([np.around(max_sharpe_port['x']*100,2), 
                        np.around(portfolio_stats(max_sharpe_port['x']),4)])
min_vol_port_data = np.concatenate([around(min_vol_port['x']*100,2), 
                     around(portfolio_stats(min_vol_port['x']),4)])
ret_10_port_data = np.concatenate([around(100*efport.iloc[0,3],2),
                    around(portfolio_stats(efport.iloc[0,3]),4)])
bl_port_data = np.concatenate([around(bl_port_wts.flatten(),2),
                    around(portfolio_stats(bl_port_wts.flatten()),4)])

test_df = pd.DataFrame({"Max Sharpe Ratio": max_sharpe_port_data,
                        "Min Volatility": min_vol_port_data,
                        "10% Returns": ret_10_port_data,
                        "Black-Litterman": bl_port_data})
test_df.index = tickers + stats

# 2021-2023 backtesting
# Prices
max_sharpe_prices = backtest_prices.multiply(test_df.iloc[0:5,0]/100, axis = 1).sum(axis = 1)
min_vol_prices = backtest_prices.multiply(test_df.iloc[0:5,1]/100, axis = 1).sum(axis = 1)
ret_10_prices = backtest_prices.multiply(test_df.iloc[0:5,2]/100, axis = 1).sum(axis = 1)
equal_wts_prices = backtest_prices.multiply(np.ones(5)/5, axis = 1).sum(axis = 1)
bl_prices = backtest_prices.multiply(test_df.iloc[0:5,3]/100, axis = 1).sum(axis = 1)

backtest_port_prices = pd.DataFrame({"Max Sharpe Ratio": max_sharpe_prices,
                        "Min Volatility": min_vol_prices,
                        "10% Returns": ret_10_prices,
                        "Equal Weights": equal_wts_prices,
                        "Black-litterman": bl_prices,
                        "S&P 500": spx_mxn})

# Summary statistics
annual_returns = backtest_port_prices.resample('Y').last().pct_change().dropna()
two_year_returns = backtest_port_prices.resample('2Y').last().pct_change().dropna()
backtest_returns = backtest_port_prices.pct_change().dropna()
backtest_means = backtest_returns.mean()
backtest_sds = backtest_returns.std()
backtest_skews = backtest_returns.skew()
backtest_kurtosis = backtest_returns.kurtosis()
backtest_VaR = backtest_returns.quantile(0.05)
backtest_cVaR = backtest_returns.apply(calcular_cvar,args = (0.95,),axis=0)
backtest_sharpes = backtest_returns.apply(sharpe_ratio, args = (rf/252,), axis = 0)
backtest_sortinos = backtest_returns.apply(sortino_ratio, args = (rf/252,), axis = 0)
backtest_drawdons = backtest_returns.apply(drawdon, axis=0)
backtest_cum_ret = (backtest_returns+1).cumprod()

backtest_summary_df = pd.DataFrame([backtest_means,backtest_sds,
                                    backtest_skews,backtest_kurtosis,
                                    backtest_VaR,backtest_cVaR,backtest_sharpes,
                                    backtest_sortinos,backtest_drawdons, 
                                    backtest_cum_ret.iloc[-1]],
                                    index = ['mean','sd','skew','kurtosis',
                                             'VaR 95%','cVaR 95%', 
                                             'sharpe ratio','sortino ratio',
                                             'max drawdon', 'cumulative returns'])

backtest_summary_df = pd.DataFrame(np.vstack([backtest_summary_df, annual_returns, two_year_returns]))
backtest_summary_df.columns = ['Max Sharpe Ratio', 'Min Volatility', '10% Returns', 'Equal Weights', 'Black-Litterman', 'S&P 500']
backtest_summary_df.index = ['mean','sd','skew','kurtosis','VaR 95%','cVaR 95%', 
                             'sharpe ratio','sortino ratio','max drawdon', 'cumulative returns',
                             '2022 annual returns','2023 annual returns', 'total returns']

# Asset Visualization df
# S&P500 en pesos
spx_all = download_data("^GSPC", inicio, fin)
# calculating assets prices in mexican pesos
spx_all = spx_all.join(df_mxn)
spx_all = spx_all['^GSPC']
spx_all_mxn = spx_all.mul(mxn, axis = 0)
spx_all_mxn.columns = ['S&P 500']
df_final = df_mxn
df_final['S&P 500'] = spx_all_mxn
df_final = df_final.apply(lambda x: 100*(x/x[0]))

# Visualizacion

# Crear pestañas
tab1, tab2 = st.tabs(["Asset Analysis", "Portfolio Analysis"])

with tab1:
  st.header("Individual Asset Analysis")
  selected_asset = st.selectbox("Select an asset to analyze:", tickers)

  st.write(f"Daily Summary Statistics for the returns of the {selected_asset} ETF between 2010 and 2020.")
  
  col1, col2, col3 = st.columns(3)
  col1.metric("Mean Returns", f"{100*summary_df.loc['mean',selected_asset]:.2%}")
  col2.metric("Volatility", f"{summary_df.loc['sd',selected_asset]:.4f}")
  col3.metric("Skew", f"{summary_df.loc['skew',selected_asset]:.2f}")
  
  col4, col5, col6 = st.columns(3)
  col4.metric("Kurtosis", f"{summary_df.loc['kurtosis',selected_asset]:.2f}")
  col5.metric("VaR 95%", f"{summary_df.loc['VaR 95%',selected_asset]:.2%}")
  col6.metric("cVaR 95%", f"{summary_df.loc['cVaR 95%',selected_asset]:.2%}")
  
  
  col7, col8, col9 = st.columns(3)
  col7.metric("Sharpe Ratio", f"{summary_df.loc['sharpe ratio',selected_asset]:.2f}")
  col8.metric("Sortino Ratio", f"{summary_df.loc['sortino ratio',selected_asset]:.2f}")
  col9.metric("Max Drawdon", f"{summary_df.loc['max drawdon',selected_asset]:.2f}")
        

  fig_asset = go.Figure()

  fig_asset.add_trace(go.Scatter(x=df_final.index, y=df_final[selected_asset], name = selected_asset))
  fig_asset.add_trace(go.Scatter(x=df_final.index, y=df_final['S&P 500'], name='S&P 500'))
  
  fig_asset.update_layout(
      title=f'Normalized Prices: {selected_asset} vs S&P 500 (Base 100)',
      xaxis_title='Date',
      yaxis_title='Normalized Price')

  st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")


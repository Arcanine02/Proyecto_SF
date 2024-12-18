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

@ st.cache_data
def asset_info():
 # Info about the assets
 cadena1 = ''' * Managed by BlackRock, the iShares Core U.S. Aggregate Bond ETF (AGG) is a U.S. fixed income ETF.
 * It seeks to track the investment results of an index composed of the total U.S. investment-grade bond market and its price is expressed in USD.
 * Its main assets are allocated on:
  United States Treasury (44.23%),
  Federal National Mortgage Association I (11.04%) And II (6.05%),
  Federal Home Loan Mortgage Corporation (5.57%),
  and Uniform MBS (1.54%).
  * For more information visit https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf.'''
 
 cadena2 = ''' * Managed by BlackRock, the iShares J.P. Morgan USD Emerging Markets Bond ETF is (as given by its name) an Emerging Markets Bond ETF 
 that exchanged in NASDAQ seeks to track the investment results of an index composed of U.S. dollar-denominated, emerging market bonds.
 * Its benchmark index is the J.P. Morgan EMBI Global Core Index.
 * This ETF is mostly formed by sovereign assets (85%) from over 30 countries such as Saudi Arabia (5.85%), Mexico (5.67%), 
 Turkey (4.92%), United Arab Emirates (4.68%), Indonesia (4.67%).
 * For more information visit https://www.ishares.com/us/products/239572/ishares-jp-morgan-usd-emerging-markets-bond-etf.'''
 
 
 cadena3 = ''' * Managed by SPDR (and by extension S&P), the SPDR Gold Shares (NYSEArca) ETF seeks to track the returns of the gold asset.
 * It is a relatively low priced ETF and is the largest physically backed gold ETF in the world.
 * For more information visit https://www.spdrgoldshares.com/.'''
 
 
 cadena4 = ''' * Managed by Invesco, the QQQ ETF is an equity ETF that seeks to track the returns of the Nasdaq 100 index.
 * It is exchanged in the NASDAQ stock market, and its assets are mostly allocated on the technology sector (59.78%), with stocks of
 Apple (8.83%), NVIDIA(8.23%), Microsoft (7.67%), Amazon (5.36%), ane Meta (5.1%) among others.
 * It also holds assets on other sectors, like Consumer Discretionary (18.28%) and Healthcare (6.05%).
 * For more information visit https://www.invesco.com/qqq-etf/en/about.html.'''
 
 
 cadena5 = '''* Managed by SPDR (and by extension S&P), the SPDR Portfolio Emerging Markets ETF is an emerging markets equity ETF 
 that tracks the returns of the S&P Emerging BMI Index.
 * Its exchanged in the NYSE ARCA and its holdings include stocks from companies like
 Taiwan Semiconductor Manufacturing (8.6%), Tencent Holdings (3.51%) and Alibaba(1.86%).
 * For more information visit https://www.ssga.com/us/en/intermediary/etfs/spdr-portfolio-emerging-markets-etf-spem.'''
 
 descriptions = [cadena1, cadena2, cadena3, cadena4, cadena5]
 
 
 descriptions_dict = zip(tickers, descriptions)
 descriptions_dict = dict(descriptions_dict)
 return descriptions_dict

descriptions_dict = asset_info()

# function to download data from yahoofinance
@st.cache_data
def download_data(assets, start_date, end_date):
  data = yf.download(assets, start = start_date, end = end_date)["Close"]
  return data

# Función para crear histogramas 
@st.cache_data
def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    # Crear el histograma base
    fig = go.Figure()
    
    # Calcular los bins para el histograma
    counts, bins = np.histogram(returns, bins=50)
    
    # Separar los bins en dos grupos: antes y después del VaR
    mask_before_var = bins[:-1] <= var_95
    
    # Añadir histograma para valores antes del VaR (rojo)
    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Returns < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))
    
    # Añadir histograma para valores después del VaR (azul)
    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Returns > VaR',
        marker_color='rgba(31, 119, 180, 0.6)'
    ))
    
    # Añadir líneas verticales para VaR y CVaR
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=2, dash='dot')
    ))
    
    # Actualizar el diseño
    fig.update_layout(
        title=title,
        xaxis_title='Returns',
        yaxis_title='Frequence',
        showlegend=True,
        barmode='overlay',
        bargap=0
    )
    
    return fig

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
# risk free rate: 1 year treasuries. 4.297, updates december 2nd 2024.
rf = 0.04297

# Expected shortfall function
@st.cache_data
def calcular_cvar(x, alpha):
 VaR = np.quantile(x,1-alpha)
 cVaR = x[x.lt(VaR)].mean()
 return(cVaR)

# sharpe ratio function
@st.cache_data
def sharpe_ratio(x, rf):
 dif = x-rf
 return(dif.mean()/dif.std())

# sortino ratio function
@st.cache_data
def sortino_ratio(x, rf):
 dif = x-rf
 return(dif.mean()/dif[dif<0].std())

# max drawdon 
@st.cache_data
def drawdon(x):
 cum_returns = (1+x).cumprod()
 max_cum_return = cum_returns.cummax()
 drawdown = (max_cum_return - cum_returns)/max_cum_return
 max_drawdown = drawdown.max()
 return(max_drawdown)
  
@st.cache_data
def assets_stats():
 means = returns_test.mean()
 sds = returns_test.std()
 skews = returns_test.skew()
 kurtosis_excess = returns_test.kurtosis()
 VaRs = returns_test.quantile(0.05)
 cVaRs = returns_test.apply(calcular_cvar,args = (0.95,),axis=0)
 sharpes = returns_test.apply(sharpe_ratio, args = (rf/252,), axis = 0)
 sortinos = returns_test.apply(sortino_ratio, args = (rf/252,), axis = 0)
 drawdowns = returns_test.apply(drawdon, axis=0)
 
 # joining all summary statistics in a dataframe
 summary_df = pd.DataFrame([means,sds,skews,kurtosis_excess,VaRs,cVaRs,sharpes,sortinos,drawdowns],
                           index = ['mean','sd','skew','kurtosis','VaR 95%','cVaR 95%', 'sharpe ratio','sortino ratio','max drawdon'])
 return summary_df

summary_df = assets_stats()

# Markowitz
# initial weights for every asset: equal weights.
initial_wts = (np.ones(n)/n)[:,newaxis]

# annualized daiy returns.
ret_an = array(returns.mean()*252)[:,newaxis]

# a function that will calculate annualized returns, volatility and sharpe ratio for the portfolio
@st.cache_data
def portfolio_stats(weights):

    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns_test.mean() * 252)[:,newaxis]
    port_vols = sqrt(multi_dot([weights.T, returns_test.cov() * 252, weights]))

    return array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Max Sharpe Ratio Portfolio
# Maximizing sharpe ratio by minimizing the negative sharpe ratio

@st.cache_data
def neg_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]
 
@ st.cache_data
def max_sharpe_portfolio():

 # Specify constraints and bounds
 cons_max_sharpe = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
 bnds_max_sharpe = tuple((0, 1) for x in range(n))
 initial_wts_max_sharpe = np.ones(n)/n
 # Optimizing portfolio
 max_sharpe_port = sco.minimize(neg_sharpe_ratio, initial_wts_max_sharpe, method = 'SLSQP', 
                                bounds = bnds_max_sharpe, constraints = cons_max_sharpe)
 return max_sharpe_port

max_sharpe_port = max_sharpe_portfolio()

# Portfolio weights
max_sharpe_port_wts = list(zip(tickers, around(max_sharpe_port['x']*100,2)))
# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
max_sharpe_port_stats = list(zip(stats, around(portfolio_stats(max_sharpe_port['x']),4)))

# Minimum Volatility Portfolio
@st.cache_data
def min_volatility(weights):
    return portfolio_stats(weights)[1]

# Specify constraints and bounds
@st.cache_data
def min_vol_portfolio():

 cons_min_vol = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
 bnds_min_vol = tuple((0, 1) for x in range(n))
 initial_wts_min_vol = np.ones(n)/n
 # Optimizing portfolio
 min_vol_port = sco.minimize(min_volatility, initial_wts_min_vol, method = 'SLSQP', 
                             bounds = bnds_min_vol, constraints = cons_min_vol)
 return min_vol_port

min_vol_port = min_vol_portfolio()
 
# Portfolio weights
min_vol_port_wts = list(zip(tickers, around(min_vol_port['x']*100,2)))
# Portfolio stats
min_vol_port_stats = list(zip(stats, around(portfolio_stats(min_vol_port['x']),4)))

# Efficient Frontier
# Minimize the volatility
@st.cache_data
def efficient_frontier():
 # Efficient frontier params
 targetrets = linspace(0.08,0.23,20)
 tvols = []
 weights_list = []
 
 for tr in targetrets:
  initial_wts_min_vol = np.ones(n)/n
  bnds_min_vol = tuple((0, 1) for x in range(n))
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
     'weights': weights_list})
 
 return efport
 
efport = efficient_frontier()

# Extracting weights and stats for the 10% return portfolio
ret_10_port_wts = list(zip(tickers, around(100*efport.iloc[0,3],2)))
ret_10_port_stats = list(zip(stats, around(portfolio_stats(efport.iloc[0,3]),4)))

# Backtesting
# S&P500 en pesos
@ st.cache_data
def sp_pesos():
 spx = download_data("^GSPC", "2021-01-01", fin)
 
 # calculating assets prices in mexican pesos
 spx = spx.join(df)
 spx = spx['^GSPC']
 
 spx_mxn = spx.mul(mxn, axis = 0)["2021-1-01":"2023-12-31"]
 return spx_mxn

spx_mxn = sp_pesos()

# All portfolios and benchmark dataframe
backtest_prices = df_mxn["2021-01-01":"2023-12-31"]

# Black-Litterman
@st.cache_data
def bl_portfolio():
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
 
 bl_port_wts = (((1/risk_aversion)*np.linalg.inv(returns_cov))@posterior_mean)/100
 return bl_port_wts

bl_port_wts = bl_portfolio()

# B-L Perspectives
# Returns
bl_returns = [0.1,-0.08,0.05,0.2,0.1]
bl_returns_dict = dict(zip(tickers, bl_returns))

# Opinions

opinion_1 = ''' * Let's begin by talking about my perspective for the iShares Core U.S. Aggregate Bond ETF (AGG): A 10% return one year from now.
My main reason for this optimistic view is fueled by Donald Trump's promises to apply tariffs to foreign merchandises.
This will increase inflation in the U.S.,  boosting the returns of fixed income assets such as Treasuries, which represent a big chunk of
the AGG ETF's asset allotacion.'''

opinion_2 = ''' * I forecast a -8% return for the iShares J.P. Morgan USD Emerging Markets Bond ETF one year from now.
My reasoing is also guided by Donald Trump's tariff policies. 
His internal strenghtening agenda will result detrimental for emerging countries, such as Mexico. 
Even with a possible rise in therse countries inflation, 
their soverign bonds won't be as attractive as U.S. treasuries, which will offer high yields accroding to my previous perspective.'''

opinion_3 = ''' * My views regarding the SPDR Gold Shares (NYSEArca) ETF converge in considering an annual freturn of 5%.
Gold is a good asset for protection against inflation. Given my pesimistic views on inflation one year from now, 
I consider gold to be a great asset to invest on. 
However, given my perspectives on both american and emerging equity (which will be discussed later), I don't consider gold to be the 
best asset in this portfolio.'''

opinion_4 = ''' * Let's delve into  what I consider to be the most promissing asset in this portfolio: The Invesco QQQ ETF, for which I
predict a 20% annual return. 
My faith in this ETF comes from both the fact that it consistently outperforms the S&P 500, and the effect of Trump's promise to cut
corporations taxes: An increase in the yields given by american equity. 
These reasons, combined with the fact that the QQQ's allocation is mostly centered on technological companies, which will only continue 
to benefit with the rise of AI, make me think that the QQQ is tis portfolio's most promising ETF.'''

opinion_5 = ''' * Finally, let's discuss my views on the SPDR Portfolio Emerging Markets ETF, for which I predict a 10% annual return.
In contrast to the detrimental effect Trump will have on emerging countries sovereign bonds, 
I don't see this ETF being greatly affected by his agenda, 
given the fact that a big part of this ETF is semiconductors, 
a neccesary good for the development of AI, and for the production of techonology in general.'''


# 2010-2020 portfolio summary
@st.cache_data
def prior_portfolio_summary():
 max_sharpe_port_data = np.concatenate([np.around(max_sharpe_port['x']*100,2), 
                         np.around(portfolio_stats(max_sharpe_port['x']),4)])
 min_vol_port_data = np.concatenate([around(min_vol_port['x']*100,2), 
                      around(portfolio_stats(min_vol_port['x']),4)])
 ret_10_port_data = np.concatenate([around(efport.iloc[0,3]*100,2),
                     around(portfolio_stats(efport.iloc[0,3]),4)])
 bl_port_data = np.concatenate([around(bl_port_wts.flatten()*100,2),
                     around(portfolio_stats(bl_port_wts.flatten()),4)])
 
 test_df = pd.DataFrame({"Max Sharpe Ratio": max_sharpe_port_data,
                         "Min Volatility": min_vol_port_data,
                         "10% Returns": ret_10_port_data,
                         "Black-Litterman": bl_port_data})
 test_df.index = tickers + stats
 return test_df

test_df = prior_portfolio_summary()

markowitz_ports = test_df.columns[0:3]

# 2021-2023 backtesting
# Summary statistics
@st.cache_data
def backtest_stats():
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
 return backtest_summary_df

backtest_summary_df = backtest_stats()

@st.cache_data
def pre_df(x):
 return x["2010-01-01":"2020-12-31"]
@st.cache_data
def post_df(x):
 return x["2021-01-01":"2023-12-31"]



# Asset Visualization df
# S&P500 en pesos
@st.cache_data
def sp_pesos():
 spx_all = download_data("^GSPC", inicio, fin)
 # calculating assets prices in mexican pesos
 spx_all = spx_all.join(df_mxn)
 spx_all = spx_all['^GSPC']
 spx_all_mxn = spx_all.mul(mxn, axis = 0)
 spx_all_mxn.columns = ['S&P 500']
 return spx_all_mxn

spx_all_mxn = sp_pesos()

@st.cache_data
def mxn_prices_full():
 
 df_final = df_mxn
 df_final['S&P 500'] = spx_all_mxn
 df_final = df_final.apply(lambda x: 100*(x/x[0]))
 return df_final
 
df_final = mxn_prices_full()

df_final_pre = pre_df(df_final)
df_final_post = post_df(df_final)

# Full period portfolios prices
@st.cache_data
def final_portfolio_prices():
 max_sharpe_prices_final = df_final.iloc[:,0:5].multiply(test_df.iloc[0:5,0]/100, axis = 1).sum(axis = 1)
 min_vol_prices_final = df_final.iloc[:,0:5].multiply(test_df.iloc[0:5,1]/100, axis = 1).sum(axis = 1)
 ret_10_prices_final = df_final.iloc[:,0:5].multiply(test_df.iloc[0:5,2]/100, axis = 1).sum(axis = 1)
 equal_wts_prices_final = df_final.iloc[:,0:5].multiply(np.ones(5)/5, axis = 1).sum(axis = 1)
 bl_prices_final = df_final.iloc[:,0:5].multiply(test_df.iloc[0:5,3]/100, axis = 1).sum(axis = 1)
 bl_prices_final = (100*bl_prices_final)/bl_prices_final[0]
 port_prices_final = pd.DataFrame({"Max Sharpe Ratio": max_sharpe_prices_final,
                         "Min Volatility": min_vol_prices_final,
                         "10% Returns": ret_10_prices_final,
                         "Equal Weights": equal_wts_prices_final,
                         "Black-litterman": bl_prices_final,
                         "S&P 500": (100*(spx_all_mxn))/spx_all_mxn[0]})
 port_prices_final = port_prices_final.replace([0],np.nan).dropna()
 
 return port_prices_final

port_prices_final = final_portfolio_prices()


port_prices_final_pre = pre_df(port_prices_final)
port_prices_final_post = post_df(port_prices_final)

@st.cache_data
def final_portfolio_returns():
 port_returns_final = port_prices_final.pct_change().dropna()
 port_returns_final = port_returns_final.replace([np.inf, -np.inf,-1],np.nan).dropna()
 return port_returns_final

port_returns_final = final_portfolio_returns()


port_returns_final_pre = pre_df(port_returns_final)
port_returns_final_post = post_df(port_returns_final)

# Visualization

st.title("Portfolio Management App by Santiago Anaya")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Asset Analysis", "Markowitz Portfolios Analysis", "Black-Litterman Portfolio Analysis"])

with tab1:
  st.header("Individual Asset Analysis")
  st.write("Every asset in this dashboard is expressed in mexican pesos.") 
  st.write("This dashboard considers risk-free rate of 4.297%, the 1 year Treasury yield as of december 2nd 2024.")
  selected_asset = st.selectbox("Select an asset to analyze:", tickers)

  st.write(descriptions_dict[selected_asset])

  st.subheader(f"Daily Summary Statistics for the returns of the {selected_asset} ETF between 2010 and 2020.")
  
  col1, col2, col3 = st.columns(3)
  col1.metric("Mean Returns", f"{100*summary_df.loc['mean',selected_asset]:.3%}")
  col2.metric("Volatility", f"{summary_df.loc['sd',selected_asset]:.3f}")
  col3.metric("Skew", f"{summary_df.loc['skew',selected_asset]:.2f}")
  
  col4, col5, col6 = st.columns(3)
  col4.metric("Kurtosis", f"{summary_df.loc['kurtosis',selected_asset]:.2f}")
  col5.metric("VaR 95%", f"{summary_df.loc['VaR 95%',selected_asset]:.2%}")
  col6.metric("cVaR 95%", f"{summary_df.loc['cVaR 95%',selected_asset]:.2%}")
  
  
  col7, col8, col9 = st.columns(3)
  col7.metric("Sharpe Ratio", f"{summary_df.loc['sharpe ratio',selected_asset]:.3f}")
  col8.metric("Sortino Ratio", f"{summary_df.loc['sortino ratio',selected_asset]:.3f}")
  col9.metric("Max Drawdon", f"{summary_df.loc['max drawdon',selected_asset]:.3f}")
        

  fig_asset = go.Figure()

  fig_asset.add_trace(go.Scatter(x=df_final.index, y=df_final[selected_asset], name = selected_asset))
  fig_asset.add_trace(go.Scatter(x=df_final.index, y=df_final['S&P 500'], name='S&P 500'))
  
  fig_asset.update_layout(
      title=f'Normalized Prices: {selected_asset} vs S&P 500 (Base 100)',
      xaxis_title='Date',
      yaxis_title='Normalized Price')

  st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")
 
  # Histogram for VaR and cVaR
  hist_returns = returns_test[selected_asset]
  hist_fig = crear_histograma_distribucion(hist_returns,
                                           np.quantile(hist_returns,0.05) , 
                                           calcular_cvar(hist_returns,0.95), 
                                           f"Daily returns of {selected_asset} between 2010 and 2020")
 
  st.plotly_chart(hist_fig, use_container_width=True, key="returns_hist")

with tab2:
 st.header("Optimal Portfolios")

 st.write('''This plot represents the curve of optimal portfolios given by the Markowitz model, known as the efficient frontier.
 The dots in the plot represents the optimal portfolios in terms of volatility, when considering a fixed level of returns.
 This means that if one wished to recieve a certain level of returns, the least possible level of volatility given by the 
 Markowitz model is shown in the plot.
 The stars in the plot represent both the portfolio with the minumum volatility (green star) and the portfolio with the maximum 
 sharpe ratio (red star). Below this chart, the user will be able to select which portfolio they wish to study, and 
 an exploratory analysis of said portfolio will be shown.''')

 # Plot efficient frontier portfolio
 fig_ef = px.scatter(
   efport, x='targetvols', y='targetrets',  color='targetsharpe', 
   range_color = [0.8,1.2],
   labels={'targetrets': 'Expected Return', 'targetvols': 'Expected Volatility','targetsharpe': 'Sharpe Ratio'},
   title="Efficient Frontier Portfolio"
    ).update_traces(mode='markers', marker=dict(symbol='cross'))
  
 # Plot maximum sharpe portfolio
 fig_ef.add_scatter(
     mode='markers',
     x=[100*portfolio_stats(max_sharpe_port['x'])[1]],
     y=[100*portfolio_stats(max_sharpe_port['x'])[0]],
     marker=dict(color='red', size=20, symbol='star'),
     name = 'Max Sharpe'
 ).update(layout_showlegend=False)
  
 # Plot minimum variance portfolio
 fig_ef.add_scatter(
     mode='markers',
     x=[100*portfolio_stats(min_vol_port['x'])[1]],
     y=[100*portfolio_stats(min_vol_port['x'])[0]],
     marker=dict(color='green', size=20, symbol='star'),
     name = 'Min Variance'
 ).update(layout_showlegend=False)
  
 # Show spikes
 fig_ef.update_xaxes(showspikes=True)
 fig_ef.update_yaxes(showspikes=True)
 st.plotly_chart(fig_ef, use_container_width=True, key="efficient_frontier")

 # Select portfolio
 selected_portfolio = st.selectbox("Select a portfolio to analyze:", markowitz_ports)
 st.subheader("Weights of the assets in the portfolio")
 col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
 col_w1.metric(tickers[0], f"{0.01*test_df.loc[tickers[0],selected_portfolio]:.2%}")
 col_w2.metric(tickers[1], f"{0.01*test_df.loc[tickers[1],selected_portfolio]:.2%}")
 col_w3.metric(tickers[2], f"{0.01*test_df.loc[tickers[2],selected_portfolio]:.2%}")
 col_w4.metric(tickers[3], f"{0.01*test_df.loc[tickers[3],selected_portfolio]:.2%}")
 col_w5.metric(tickers[4], f"{0.01*test_df.loc[tickers[4],selected_portfolio]:.2%}")
 
 st.subheader("2010-2020 Portfolio Construction")
 st.write('''This section will show an exploratory analysis of the chosen portfolio, considering the time period 
 for which the portfolio was built. Bearing this in mind, the user can consider the below ststistics as those balonging to the 
 best possible porfolio given the restriction they selected (such as minimum volatility or maximum sharpe ratio).''')
 
 st.subheader("Statistics of the selected portfolio's daily returns")
 col_m1, col_m2, col_m3 = st.columns(3)
 col_m1.metric("Mean", f"{test_df.loc["Returns",selected_portfolio]:.3%}")
 col_m2.metric("Volatility", f"{test_df.loc["Volatility",selected_portfolio]:.3f}")
 col_m3.metric("Sharpe Ratio", f"{test_df.loc["Sharpe Ratio",selected_portfolio]:.3f}")

 # Plotting the portfolio vs S&P 500 benchmark
 fig_port1 = go.Figure()

 fig_port1.add_trace(go.Scatter(x=port_prices_final_pre.index, y=port_prices_final_pre[selected_portfolio], name = selected_portfolio))
 fig_port1.add_trace(go.Scatter(x=df_final_pre.index, y=df_final_pre['S&P 500'], name='S&P 500'))
  
 fig_port1.update_layout(
     title=f'Normalized Prices: {selected_portfolio} vs S&P 500 (Base 100)',
     xaxis_title='Date',
     yaxis_title='Normalized Price')

 st.plotly_chart(fig_port1, use_container_width=True, key="price_normalized_port")
 
 # Histogram for VaR and cVaR
 hist_returns_port = port_returns_final_pre[selected_portfolio]
 hist_fig_port = crear_histograma_distribucion(hist_returns_port,
                                          np.quantile(hist_returns_port,0.05) , 
                                          calcular_cvar(hist_returns_port,0.95), 
                                          f"Daily returns of {selected_portfolio} between 2010 and 2020")
 
 st.plotly_chart(hist_fig_port, use_container_width=True, key="returns_hist_port")

 
 st.subheader("2021-2023 backtesting")
 
 st.write('''These portfolios were designed considering the time period between 2010 and 2020. 
 As such, the asset allocation is heavily dependent on the economic climate present during that timeframe. 
 This section was created with the objective to show the importance of rebalancing the portfolio, as the economic situation lived
 between 2010 and 2020 was extremely different than the one lived between 2021 and 2023.
 From here on out, I will show the descriptive stats of the optimal portfolios made considering the 2010-2020 period, 
 when exposed to the 2021-2023 macroeconomic conditions.''')

 st.subheader("Statistics of the selected portfolio's daily returns")
 col_m1b, col_m2b, col_m3b = st.columns(3)
 col_m1b.metric("Mean", f"{backtest_summary_df.loc["mean",selected_portfolio]:.3%}")
 col_m2b.metric("Volatility", f"{backtest_summary_df.loc["sd",selected_portfolio]:.3f}")
 col_m3b.metric("Skew", f"{backtest_summary_df.loc["skew",selected_portfolio]:.2f}")
 
 col_m4b, col_m5b, col_m6b = st.columns(3)
 col_m4b.metric("Kurtosis", f"{backtest_summary_df.loc["kurtosis",selected_portfolio]:.2f}")
 col_m5b.metric("VaR 95%", f"{backtest_summary_df.loc["VaR 95%",selected_portfolio]:.2%}")
 col_m6b.metric("cVaR 95%", f"{backtest_summary_df.loc["cVaR 95%",selected_portfolio]:.2%}")
 
 col_m7b, col_m8b, col_m9b = st.columns(3)
 col_m7b.metric("Sharpe Ratio", f"{backtest_summary_df.loc["sharpe ratio",selected_portfolio]:.3f}")
 col_m8b.metric("Sortino Ratio", f"{backtest_summary_df.loc["sortino ratio",selected_portfolio]:.3f}")
 col_m9b.metric("Max Drawdon", f"{backtest_summary_df.loc["max drawdon",selected_portfolio]:.3f}")
 
 col_m10b, col_m11b, col_m12b = st.columns(3)
 col_m10b.metric("2022 Returns", f"{backtest_summary_df.loc["2022 annual returns",selected_portfolio]:.2%}")
 col_m11b.metric("2023 Returns", f"{backtest_summary_df.loc["2023 annual returns",selected_portfolio]:.2%}")
 col_m12b.metric("Total Cumulative Returns", f"{backtest_summary_df.loc["total returns",selected_portfolio]:.2%}")

 # Plotting the portfolio vs S&P 500 benchmark
 fig_port2 = go.Figure()

 fig_port2.add_trace(go.Scatter(x=port_prices_final_post.index, y=port_prices_final_post[selected_portfolio], name = selected_portfolio))
 fig_port2.add_trace(go.Scatter(x=df_final_post.index, y=df_final_post['S&P 500'], name='S&P 500'))
  
 fig_port2.update_layout(
     title=f'Normalized Prices: {selected_portfolio} vs S&P 500 (Base 100)',
     xaxis_title='Date',
     yaxis_title='Normalized Price')

 st.plotly_chart(fig_port2, use_container_width=True, key="price_normalized_port_2")
 
 # Histogram for VaR and cVaR
 hist_returns_port_2 = port_returns_final_post[selected_portfolio]
 hist_fig_port_2 = crear_histograma_distribucion(hist_returns_port_2,
                                          np.quantile(hist_returns_port_2,0.05) , 
                                          calcular_cvar(hist_returns_port_2,0.95), 
                                          f"Daily returns of {selected_portfolio} between 2021 and 2023")
 
 st.plotly_chart(hist_fig_port_2, use_container_width=True, key="returns_hist_port_2")

with tab3:
 
 st.header("Black-Litterman Portfolio Analysis")

 st.subheader("Investor views for each asset")
 st.write(opinion_1)
 st.write(opinion_2)
 st.write(opinion_3)
 st.write(opinion_4)
 st.write(opinion_5)
 
 st.subheader("Weights of the assets in the portfolio")
 col_wb1, col_wb2, col_wb3 = st.columns(3)
 col_wb1.metric(tickers[0], f"{0.01*test_df.loc[tickers[0],"Black-Litterman"]:.2%}")
 col_wb2.metric(tickers[1], f"{0.01*test_df.loc[tickers[1],"Black-Litterman"]:.2%}")
 col_wb3.metric(tickers[2], f"{0.01*test_df.loc[tickers[2],"Black-Litterman"]:.2%}")


 col_wb4, col_wb5, col_wb6 = st.columns(3)
 col_wb4.metric(tickers[3], f"{0.01*test_df.loc[tickers[3],"Black-Litterman"]:.2%}")
 col_wb5.metric(tickers[4], f"{0.01*test_df.loc[tickers[4],"Black-Litterman"]:.2%}")
 col_wb6.metric("Risk-Free", f"{(1 - 0.01*test_df.loc[tickers[0:5],"Black-Litterman"].sum()):.2%}")
 
 st.subheader("2010-2020 Hypothetic Portfolio Performance")

 st.write('''While it may be true that this portfolio was built considering its performance during the year 2025, 
 I believe its important to at least analyze its performance when comparing it to the other portfolios.
 As such, I will show its statistics in both timeframes: 2010-2020 and 2021-2023.''')
 st.subheader("Statistics of the selected portfolio's daily returns")
 col_mb1, col_mb2, col_mb3 = st.columns(3)
 col_mb1.metric("Mean", f"{test_df.loc["Returns","Black-Litterman"]:.3%}")
 col_mb2.metric("Volatility", f"{test_df.loc["Volatility","Black-Litterman"]:.3f}")
 col_mb3.metric("Sharpe Ratio", f"{test_df.loc["Sharpe Ratio","Black-Litterman"]:.3f}")

 # Plotting the portfolio vs S&P 500 benchmark
 fig_port1_bl = go.Figure()

 fig_port1_bl.add_trace(go.Scatter(x=port_prices_final_pre.index, y=port_prices_final_pre["Black-litterman"], name = "Black-Litterman"))
 fig_port1_bl.add_trace(go.Scatter(x=df_final_pre.index, y=df_final_pre['S&P 500'], name='S&P 500'))
  
 fig_port1_bl.update_layout(
     title= "Normalized Prices: Black-Litterman vs S&P 500 (Base 100)",
     xaxis_title='Date',
     yaxis_title='Normalized Price')

 st.plotly_chart(fig_port1_bl, use_container_width=True, key="price_normalized_port_bl")
 
 # Histogram for VaR and cVaR
 hist_returns_port_bl = port_returns_final_pre["Black-litterman"]
 hist_fig_port_bl = crear_histograma_distribucion(hist_returns_port_bl,
                                          np.quantile(hist_returns_port_bl,0.05) , 
                                          calcular_cvar(hist_returns_port_bl,0.95), 
                                          "Daily returns of Black-Litterman between 2010 and 2020")
 
 st.plotly_chart(hist_fig_port_bl, use_container_width=True, key="returns_hist_port_bl")

 
 st.subheader("2021-2023 Hypothetic Portfolio Performance")

 st.write('''Just as in the Markowitz section, from this point onward, I'll show the descriptive stats of the 
 Black-Litterman portfolio designed by considering the economic climate of the year 2025. 
 This is more of an experiment than a backtest, as the performance of this portfolio hasn't been measured for its intended timeframe.
 However, it will be interesting to analyse its performance during the post-pandemic years.''')
 
 st.subheader("Statistics of the selected portfolio's daily returns")
 col_m1bl, col_m2bl, col_m3bl = st.columns(3)
 col_m1bl.metric("Mean", f"{backtest_summary_df.loc["mean","Black-Litterman"]:.3%}")
 col_m2bl.metric("Volatility", f"{backtest_summary_df.loc["sd","Black-Litterman"]:.3f}")
 col_m3bl.metric("Skew", f"{backtest_summary_df.loc["skew","Black-Litterman"]:.2f}")
 
 col_m4bl, col_m5bl, col_m6bl = st.columns(3)
 col_m4bl.metric("Kurtosis", f"{backtest_summary_df.loc["kurtosis","Black-Litterman"]:.2f}")
 col_m5bl.metric("VaR 95%", f"{backtest_summary_df.loc["VaR 95%","Black-Litterman"]:.2%}")
 col_m6bl.metric("cVaR 95%", f"{backtest_summary_df.loc["cVaR 95%","Black-Litterman"]:.2%}")
 
 col_m7bl, col_m8bl, col_m9bl = st.columns(3)
 col_m7bl.metric("Sharpe Ratio", f"{backtest_summary_df.loc["sharpe ratio","Black-Litterman"]:.3f}")
 col_m8bl.metric("Sortino Ratio", f"{backtest_summary_df.loc["sortino ratio","Black-Litterman"]:.3f}")
 col_m9bl.metric("Max Drawdon", f"{backtest_summary_df.loc["max drawdon","Black-Litterman"]:.3f}")
 
 col_m10bl, col_m11bl, col_m12bl = st.columns(3)
 col_m10bl.metric("2022 Returns", f"{backtest_summary_df.loc["2022 annual returns","Black-Litterman"]:.2%}")
 col_m11bl.metric("2023 Returns", f"{backtest_summary_df.loc["2023 annual returns","Black-Litterman"]:.2%}")
 col_m12bl.metric("Total Cumulative Returns", f"{backtest_summary_df.loc["total returns","Black-Litterman"]:.2%}")

 # Plotting the portfolio vs S&P 500 benchmark
 fig_port2_bl = go.Figure()

 fig_port2_bl.add_trace(go.Scatter(x=port_prices_final_post.index, y=port_prices_final_post["Black-litterman"], name = "Black-Litterman"))
 fig_port2_bl.add_trace(go.Scatter(x=df_final_post.index, y=df_final_post['S&P 500'], name='S&P 500'))
  
 fig_port2_bl.update_layout(
     title=f'Normalized Prices: Black-Litterman vs S&P 500 (Base 100)',
     xaxis_title='Date',
     yaxis_title='Normalized Price')

 st.plotly_chart(fig_port2_bl, use_container_width=True, key="price_normalized_port_2_bl")
 
 # Histogram for VaR and cVaR
 hist_returns_port_2_bl = port_returns_final_post["Black-litterman"]
 hist_fig_port_2_bl = crear_histograma_distribucion(hist_returns_port_2_bl,
                                          np.quantile(hist_returns_port_2_bl,0.05) , 
                                          calcular_cvar(hist_returns_port_2_bl,0.95), 
                                          "Daily returns of Black-Litterman between 2021 and 2023")
 
 st.plotly_chart(hist_fig_port_2_bl, use_container_width=True, key="returns_hist_port_2_bl")

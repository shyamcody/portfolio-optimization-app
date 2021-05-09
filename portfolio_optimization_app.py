#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:05:56 2021

@author: shyambhu.mukherjee
Resources: 
https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html
https://github.com/robertmartin8/PyPortfolioOpt/tree/master/tests/resources
https://stackoverflow.com/questions/23464138/downloading-and-accessing-data-from-github-python
https://towardsdatascience.com/automating-portfolio-optimization-using-python-9f344b9380b9
first issue:
https://github.com/robertmartin8/PyPortfolioOpt/issues/88
portfolio performance:
https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html
mean historical return
[0.03835,0.0689,0.20603,0.07315,0.04033,0.0,
 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01324,0.35349,0.1957,0.0,0.01082]
"""


import streamlit as st
import pandas as pd
import math
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
data_path = r'https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/master/tests/resources/stock_prices.csv'
data_stock = pd.read_csv(data_path,index_col = 0, parse_dates = [0])
data_stock = data_stock.interpolate(method = 'time')
st.title("portfolio optimization model")
st.write("This is a demo application for portfolio optimization.")
st.subheader("Choose time effect on mean")
radio = st.radio("choose your time influence",
                 options = ['exponential effect on recent time',
                            'equal effect on all time'])
if radio == 'exponential effect on recent time':
    slider = st.slider("How many last days you want to do:",180,3600,180)
    st.write("you have chosen ",slider," days")
    mu = expected_returns.ema_historical_return(data_stock,span = slider)
    sigma = risk_models.exp_cov(data_stock,span = slider)
elif radio == 'equal effect on all time':
    mu = expected_returns.mean_historical_return(data_stock)
    sigma = risk_models.sample_cov(data_stock)
else:
    mu = expected_returns.mean_historical_return(data_stock)
    sigma = risk_models.sample_cov(data_stock)
ef = EfficientFrontier(mu,sigma)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

options = list(cleaned_weights.keys())
weights = [cleaned_weights[option] for option in options]
dataframe = pd.DataFrame()
dataframe['stocks'] = options
dataframe['weight'] = weights
dataframe = dataframe[dataframe['weight']>0]
dataframe.index = [i for i in range(dataframe.shape[0])]
st.subheader("Portfolio distribution")
st.write(dataframe)
st.subheader("portfolio results")
annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
st.write("Annual return:",round(annual_return*100,2))
st.write("Annual volatility:",round(annual_volatility*100,2))
st.write("Sharpe ratio:",round(sharpe_ratio,2))
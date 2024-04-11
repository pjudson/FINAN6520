import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.linear_model import LinearRegression

# Date Variables
startDate = "1984-01-01"
endDate = "2024-01-01"

# Consumer Price Index for All Urban Consumers:
# All Items in U.S. City Average (CPIAUCSL)
# https://fred.stlouisfed.org/series/CPIAUCSL
cpi = web.DataReader("CPIAUCSL", "fred", startDate, endDate)

# Inflation, consumer prices for the United States (FPCPITOTLZGUSA)
# https://fred.stlouisfed.org/series/FPCPITOTLZGUSA
inflation = web.DataReader("FPCPITOTLZGUSA", "fred", startDate, endDate)

# Average Price: Orange Juice, Frozen Concentrate,
# 12 Ounce Can (Cost per 16 Ounces/473.2 Milliliters) in U.S. City Average (APU0000713111)
# https://fred.stlouisfed.org/series/APU0000713111
oj = web.DataReader("APU0000713111", "fred", startDate, endDate)

# Average Price: Ground Beef, 100% Beef
# (Cost per Pound/453.6 Grams) in U.S. City Average (APU0000703112)
# https://fred.stlouisfed.org/series/APU0000703112
beef = web.DataReader("APU0000703112", "fred", startDate, endDate)

# Producer Price Index by Industry:
# Gold Ore and Silver Ore Mining: Gold Ores (PCU2122212122210)
# https://fred.stlouisfed.org/series/PCU2122212122210
gold = web.DataReader("PCU2122212122210", "fred", startDate, endDate)

# Global price of Brent Crude (POILBREUSDM)
# https://fred.stlouisfed.org/series/POILBREUSDM
oil = web.DataReader("POILBREUSDM", "fred", startDate, endDate)

# Federal Funds Effective Rate (FEDFUNDS)
# https://fred.stlouisfed.org/series/FEDFUNDS
fed = web.DataReader("FEDFUNDS", "fred", startDate, endDate)

# Average Weekly Hours of All Employees, Total Private (AWHAETP)
# https://fred.stlouisfed.org/series/AWHAETP
hours = web.DataReader("AWHAETP", "fred", startDate, endDate)

# 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y)
# https://fred.stlouisfed.org/series/T10Y2Y
yieldCurve = web.DataReader("T10Y2Y", "fred", startDate, endDate)

# University of Michigan: Consumer Sentiment (UMCSENT)
# https://fred.stlouisfed.org/series/UMCSENT
sentiment = web.DataReader("UMCSENT", "fred", startDate, endDate)

# Money Supply - M2 (M2SL)
# https://fred.stlouisfed.org/series/M2SL
m2 = web.DataReader("M2SL", "fred", startDate, endDate)

# S&P 500 Monthly Close Price
sp500 = web.DataReader("^SPX", "stooq", startDate, endDate)["Close"].resample("BME").last()
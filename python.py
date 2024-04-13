import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import FinalProject.WOE as woe
from sklearn import linear_model
from sklearn.metrics import r2_score


def printOutTheCoefficients(params,coeffecients,intercept):
    tParams = params[np.newaxis].T
    tCoeffs = coeffecients.T
    total = np.concatenate([tParams,tCoeffs],axis=1)
    totalDF = pd.DataFrame(data=total)
    totalDF.to_excel("modelOutput.xlsx")
    print(totalDF)


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
sp500 = web.DataReader("^SPX", "stooq", startDate, endDate)
sp500 = sp500.drop(columns=['Open', 'High', 'Low', 'Volume'])

joinedDfs = sp500.join(m2, how="inner")
joinedDfs = joinedDfs.join(sentiment, how="inner")
joinedDfs = joinedDfs.join(yieldCurve, how="inner")
joinedDfs = joinedDfs.join(hours, how="inner")
joinedDfs = joinedDfs.join(fed, how="inner")
joinedDfs = joinedDfs.join(oil, how="inner")
# joinedDfs = joinedDfs.join(gold, how="inner") data only exists until 2017
joinedDfs = joinedDfs.join(beef, how="inner")
joinedDfs = joinedDfs.join(oj, how="inner")
# joinedDfs = joinedDfs.join(inflation, how="inner") only yearly data
joinedDfs = joinedDfs.join(cpi, how="inner")

# Beef has NaN value
joinedDfs = joinedDfs.dropna(subset=["APU0000703112"])

# Rename columns
joinedDfs.rename(columns = {"Close": "SP500", "CPIAUCSL": "CPI", "APU0000713111": "OJ",
                            "APU0000703112": "Beef", "PCU2122212122210": "Gold",
                            "POILBREUSDM": "Oil", "FEDFUNDS": "FedFunds", "AWHAETP": "HoursWorked",
                            "T10Y2Y": "Yield", "UMCSENT": "Sentiment", "M2SL": "M2"}, inplace = True)

#Correlation Check
correlation = joinedDfs.corr(numeric_only=True)
finalIV,IV = woe.data_vars(joinedDfs,joinedDfs["SP500"])
correlation.to_excel("correlation.xlsx")
IV.to_excel("IVOutput.xlsx")

#Correlation Map
#M2, Beef , CPI have high levels of correlated with each other
#so we may want to remove one or two of them
sb.heatmap(correlation)
plt.show()

# separate for results and input sets
dfResults = joinedDfs["SP500"]
dfInputs = joinedDfs.drop("SP500", axis=1)

# split between sets
inputsTrain, inputsTest, resultTrain, resultTest = train_test_split(dfInputs, dfResults,
                                                                    test_size=0.5, random_state=1)

#Linear Regression
#Since we are predicting an S&P numerical value, it would be a linear regression
linRegr = linear_model.LinearRegression()
linRegr.fit(inputsTrain,resultTrain)
Ypredict = linRegr.predict(inputsTrain)

#calculated R2
r2easy = r2_score(resultTrain,Ypredict)
print("Our calculated coeffs m:{} and b:{} and our r2 is {}".format(linRegr.coef_,linRegr.intercept_,r2easy))

#I am thinking, in order to make this variable accurate, all our our predictor variables need to be time -1
# for example, CPI data comes out and then the S&P reacts. So our R2 is really high since the S&P value is already factoring that in
# We will need to take all the values for the month prior in order to predict what next month's S&P result looks like


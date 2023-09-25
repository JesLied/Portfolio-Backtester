from sleipnir_backtester import Backtester
import matplotlib.pyplot as plt
from pprint import pprint
import yfinance
import random
import pandas as pd
import numpy as np

# tickers = ["AAPL", "GDX", "NVDA", "TSLA", "MSFT", "GOOG", "AMZN", "FB", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "XOM", "CMCSA", "VZ", "INTC", "T", "PFE", "KO", "MRK", "CSCO", "WMT", "PEP", "NFLX", "ABT", "CVX", "NKE", "ADBE", "CRM", "ABBV", "MCD", "TMO", "ACN", "COST", "MDT", "AMGN", "AVGO", "NEE", "TXN", "UNP", "LLY", "PM", "HON", "LIN", "DHR", "UPS", "SBUX", "ORCL", "LOW", "IBM", "QCOM", "AMT", "C", "BA", "MMM", "CAT", "GE", "GILD", "FIS", "INTU", "ANTM", "AMD", "CVS", "BKNG", "ISRG", "MO", "SPGI", "LMT", "CHTR", "MDLZ", "ZTS", "AXP", "BDX", "CI", "TGT", "SYK", "PLD", "CME", "DUK", "CCI", "TJX", "FISV", "TFC", "ADP", "USB", "CB", "D", "ICE", "SO", "CL", "NSC", "VRTX", "APD", "SPG", "GS", "TMO", "COP", "PNC", "CSX", "EQIX", "ILMN", "BIIB", "AON", "SCHW", "MMC", "ATVI", "SRE", "AEP", "BLK", "CLX", "A", "MS", "EL", "FDX", "CCI", "WM", "EW", "GM", "LHX", "ZTS", "SYY", "NEM", "EBAY", "PSA", "KMB", "ETN", "DOW", "WBA", "BMY", "LRCX", "AFL", "ITW", "ADSK"]

# prices = yfinance.download(" ".join(tickers), start="2014-01-01", end="2023-01-01", progress=False)["Adj Close"]
# # drop columns with all nans
# prices = prices.dropna(axis=1, how="all")
# # fill nans with previous value
# prices = prices.fillna(method="ffill")
# # if a price is more than 2 stddevs away from the mean, replace it with the mean
# prices = prices.mask(prices.sub(prices.mean()).div(prices.std()).abs().gt(2)).fillna(method="ffill")
# # drop rows with any nans
# prices = prices.dropna(axis=0)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

window = 300
mom = prices.copy()
for col in mom.columns:
    mom[col] = prices[col].ewm(span=window).mean().pct_change()
    # mom[col] = prices[col].rolling(window).mean().pct_change()

# get the peercentilescore of each stock in each day
mom = mom.rank(axis=1, pct=True)

weights = prices.copy()
i = 0 
rebace = 120
for date in weights.index:
    if i % rebace == 0:
        # number of stocks where rank is 1
        nums = 6
        # nums = len(weights.columns)
        
        row = mom.loc[date, :]
        # rank the columns based on values
        top_stocks = row.sort_values(ascending=False).index[:nums]
        # get the weights
        weights.loc[date, :] = 0
        weights.loc[date, top_stocks] = 1/nums


    else:
        # get previous weights
        weights.loc[weights.index[i], :] = weights.loc[weights.index[i-1], :]
    i += 1

p = prices.copy()
w = weights.copy()

prices = prices.to_dict(orient="index")
weights = weights.to_dict(orient="index")

bt = Backtester(cash=10**6, verbose=False
                , commission=0, slippage=0
                )

bt.run(prices, weights, progress=False)

result = bt.get_results()
print("Excess Return            : {:.2%}".format(result.excess_return))
print("Days                     : {:.0f}".format(result.days))
print("Annualised Excess Return : {:.2%}".format(result.annualised_excess_return))
print("Sharpe                   : {:.2f}".format(result.sharpe))
print("Max Drawdown             : {:.2%}".format(result.max_drawdown))
print("Calmar Ratio             : {:.2f}".format(result.calmar_ratio))
print("Orders                   : {:.0f}".format(result.orders))
print("Fees                     : {:.2%}".format(bt.commission + bt.slippage))
print("VaR (95%)                : {:.2%}".format(result.var_95))
print("VaR (99%)                : {:.2%}".format(result.var_99))


# bt.plot_results(show=True)
# print(bt.get_orders())

# portfolio = bt.portfolio_values
# cash = [portfolio[i]["cash"] for i in portfolio]
# plt.plot(cash)
# plt.show()



# TEST 1
fee = bt.commission + bt.slippage

# make a series with boolean if index should be
# traded, based on if the weights changed from last
# day
to_trade = pd.Series(False, index=w.index)
for i in range(len(w)):
    if i == 0:
        continue
    # if the weights (w) changed from last day, trade
    if not (w.iloc[i] == w.iloc[i-1]).all():
        to_trade.iloc[i] = True

positions_df = pd.DataFrame(index=w.index, columns=w.columns)
portfolio_df = pd.DataFrame(index=w.index, columns=["cash", "value", "total"])

portfolio_df.iloc[0] = [bt.cash, 0, bt.cash]
positions_df = positions_df.fillna(0)

positions_df = p.fillna(0).pct_change()
positions_df = positions_df * w

d = positions_df.apply(lambda x: list(filter(lambda v: v != 0, w.loc[x.name] * x)), axis=1)
# split d into N columns
d = pd.DataFrame(d.to_list(), index=d.index)

# # get sum for each row
# plt.plot((1 + d.sum(axis=1)).cumprod())

# plt.plot((1 + p.pct_change()).mean(axis=1).cumprod())
# plt.show()
# BACKTESTER
# GitHub @JesLied
# Created: 2023-09-19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from pprint import pprint


class Backtester():
    """
    ## Backtester

    A backtester class that trades based on given
    prices and weights. Therefore, no signal column
    is needed, all signal creations should be done
    externally.

    ### Parameters:
    - `cash`: Initial cash amount. Default: 1000000
    - `commission`: Commission per trade. Default: 0.001
    - `slippage`: Slippage per trade. Default: 0.001
    - `allow_shorting`: Allow shorting of assets. Default: False
    - `verbose`: Print logs. Default: False

    ### Methods:
    - `run`: Run the backtest
    - `get_results`: Get the results of the backtest
    - `get_metrics`: Get the metrics of the backtest
    - `get_transactions`: Get the transactions of the backtest

    ### Attributes:
    - `__version__`: Version of the backtester.

    ### Example:
    ```python
    from backtester import Backtester

    bt = Backtester(cash=1000000, commission=0.001)
    
    prices = {"2020-01-01": {"AAPL": 100, "MSFT": 200}
             , "2020-01-08": {"AAPL": 101, "MSFT": 201}
             , "2020-01-15": {"AAPL": 100, "MSFT": 202}}

    weights = {"2020-01-01": {"AAPL": 0.5, "MSFT": 0.5}
              , "2020-01-08": {"AAPL": 0.3, "MSFT": 0.8}}

    bt.run(prices, weights)

    print(bt.get_results())
    ```
    """
    __version__ = 1.1








    def __init__(self
                , cash=1000000
                , commission=0.001
                , slippage=0.001
                , allow_shorting=False
                , verbose=False):

        ########## PARAMETER ASSIGNMENT ##########
        self.cash = cash
        self.commission = commission
        self.slippage = slippage
        self.allow_shorting = allow_shorting
        self.verbose = verbose

        # {order_id: {"date": date, "asset": asset, "price": price, "quantity": quantity, "type": type}}
        self.orders = {}

        # (date: {"total": total, "cash": cash, "assets": assets})
        self.portfolio_values = {}
    
        ########## PARAMETER CHECKING ##########
        # 1 - Check the parameters are correct type
        if not isinstance(self.cash, (int, float)):
            raise TypeError("cash must be an integer or float")
        if not isinstance(self.commission, (int, float)):
            raise TypeError("commission must be an integer or float")
        if not isinstance(self.slippage, (int, float)):
            raise TypeError("slippage must be an integer or float")
        if not isinstance(self.allow_shorting, bool):
            raise TypeError("allow_shorting must be a boolean")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean")

        # 2 - Check the parameters are correct value
        if self.cash < 0:
            raise ValueError("cash must be a positive number")
        if self.commission < 0:
            raise ValueError("commission must be a positive number")
        if self.slippage < 0:
            raise ValueError("slippage must be a positive number")    




    def run(self, prices, weights, rebalance_frequency=False, progress=True):
        """
        ## Run the backtest

        ### Parameters:
        - `prices`: A dictionary or Dataframe containing the prices of assets
        - `weights`: A dictionary or Dataframe containing the weights of assets
        - `rebalance_frequency`: The frequency of rebalancing, can be turned off by False. Default: "M". 
            Possible values: "Y", "H", "Q", "M", "W", "D", False

        ### Example:
        ```python
        from backtester import Backtester

        bt = Backtester(cash=1000000, commission=0.001)

        prices = {"2020-01-01": {"AAPL": 100, "MSFT": 200}
                , "2020-01-08": {"AAPL": 101, "MSFT": 201}
                , "2020-01-15": {"AAPL": 100, "MSFT": 202}}

        weights = {"2020-01-01": {"AAPL": 0.5, "MSFT": 0.5}
                , "2020-01-08": {"AAPL": 0.3, "MSFT": 0.8}
                , "2020-01-15": {"AAPL": 0.2, "MSFT": 0.8}}
        
        bt.run(prices, weights)
        ```

        ### Returns:
        `None`
        """
        
        ########## PARAMETER CHECKING ##########
        # 1 - Check the parameters are correct type
        if not isinstance(prices, (dict, pd.DataFrame)):
            raise TypeError("prices must be a dictionary or DataFrame")
        if not isinstance(weights, (dict, pd.DataFrame)):
            raise TypeError("weights must be a dictionary or DataFrame")
        if not isinstance(rebalance_frequency, (str, bool)):
            raise TypeError("rebalance_frequency must be a string or boolean")
        
        # 2 - Check values and keys of the parameters
        if isinstance(prices, dict):
            for i in prices:
                if not isinstance(prices[i], dict):
                    raise ValueError("prices values must be dictionaries")
                for j in prices[i]:
                    if not isinstance(j, str):
                        raise ValueError("prices values keys must be strings")
                    if not isinstance(prices[i][j], (int, float)):
                        raise ValueError("prices values values must be integers or floats")

        if isinstance(weights, dict):
            for i in weights:
                if not isinstance(weights[i], dict):
                    raise ValueError("weights values must be dictionaries")
                for j in weights[i]:
                    if not isinstance(j, str):
                        raise ValueError("weights values keys must be strings")
                    if not isinstance(weights[i][j], (int, float)):
                        raise ValueError("weights values must be integers or floats")
        
        # 3 - Check the parameters have the same keys
        if isinstance(prices, dict) and isinstance(weights, dict):
            if prices.keys() != weights.keys():
                raise ValueError("prices and weights must have the same keys")
            
            for i in prices:
                if prices[i].keys() != weights[i].keys():
                    raise ValueError("prices and weights must have the same keys")

        # 4 - Check if keys can be made into datetime
        try:
            pd.to_datetime(list(prices.keys()))
        except:
            raise ValueError("prices keys must be able to be converted into datetime")

        # 5 - Check weights sum to 1
        if isinstance(weights, dict):
            for i in weights:
                if not np.isclose(sum(weights[i].values()), 1):
                    raise ValueError("weights must sum to 1")
        
        
        ########## PARAMETER ASSIGNMENT ##########
        # 1 - Convert the parameters to DataFrames
        self.prices = prices
        self.weights = weights
        self.starting_value = self.cash

        if isinstance(self.prices, dict):
            self.prices = pd.DataFrame(self.prices).T
            self.prices.index = pd.to_datetime(self.prices.index)
        if isinstance(self.weights, dict):
            self.weights = pd.DataFrame(self.weights).T
            self.weights.index = pd.to_datetime(self.weights.index)

        self.positions = pd.Series([0]*len(self.prices.columns), index=self.prices.columns) # symbol: quantity
        self.portfolio_value = self.cash
        self.cash = self.cash
        

        self.placed_orders = {} # order_id: [symbol, quantity, price, type, date]
        self.order_id = 0

        self.next_rebalance_date = self._get_next_rebalance_date_(self.prices.index[0], rebalance_frequency)
        
        ########## BACKTESTING ##########
        for i in tqdm(range(len(self.prices)), desc="Backtesting", disable=not progress):
            idx = self.prices.index[i]

            # 1 - Get the prices and weights for the current date
            current_prices = self.prices.loc[idx]
            current_weights = self.weights.loc[idx]
            previous_weights = self.weights.iloc[i - 1] if i != 0 else pd.Series(0, index=current_weights.index) # previous weights, else 0 for all weights

            # 2 - Check if there should be a rebalance
            #     - New weights
            #     - Rebalance date
            #     - First date
        
            if not current_weights.equals(previous_weights) or idx == self.next_rebalance_date or idx == self.prices.index[0]:
                    
                # 2.1 - Get the assets to close
                close_assets = self._get_close_quantity_(current_weights, previous_weights, current_prices)

                for symbol in close_assets:
                    self._close_position_(symbol, current_prices[symbol], idx)

                
                # 2.2 - Update the porfolio value after closing positions
                # self.portfolio_value = self._get_portfolio_value_(idx)

                # 2.3 - Get the assets to buy more
                buy_quantities = self._get_expansion_quantity_(current_weights, previous_weights, current_prices)

                # - 2.1 - Get the required change in quantity
                # Set previous weights to 0 if first date
                if idx == self.prices.index[0]:
                    previous_weights = pd.Series(0, index=current_weights.index)

                # if rebalance_frequency == False:
                #     required_change_in_quantity = self._get_required_change_in_quantity_(current_weights, previous_weights, current_prices)
                # else:
                #     required_change_in_quantity = self._get_required_change_in_quantity_(current_weights, actual_weights, current_prices)

                # - 2.2 - Place the orders
                # close positions first
                # for symbol, quantity in required_change_in_quantity[required_change_in_quantity < 0].items():
                #     if quantity != 0:
                #         self._close_position_(symbol, current_prices[symbol], idx)


                for symbol, quantity in buy_quantities[buy_quantities > 0].items():
                    # - 2.2.1 - Check if the side changes, if so, close the position first
                    # if np.sign(quantity) != np.sign(self.positions.get(symbol, 0)) and self.positions.get(symbol, 0) != 0:
                    #     self._close_position_(symbol, current_prices[symbol], idx)

                    # - 2.2.2 - Place the order
                    if quantity != 0:
                        self._place_order_(symbol, quantity, current_prices[symbol], idx)


            # 3 - Update values for next iteration
            self.portfolio_value = self._get_portfolio_value_(idx)
            self.next_rebalance_date = self._get_next_rebalance_date_(idx, rebalance_frequency)

            # 2 - Get the value of each asset
            asset_values = self.prices.loc[idx] * self.positions
            # 3 - Update the portfolio values
            self.portfolio_values[idx] = {"total": self.portfolio_value, "cash": self.cash, "assets": asset_values.to_dict()}





    def get_orders(self):
        """Get the orders"""

        return pd.DataFrame(self.placed_orders).T.rename(columns={0: "symbol", 1: "quantity", 2: "price", 3: "type", 4: "date"})



    def get_results(self):
        """Get the results"""

        # Get the portfolio values
        values_df = pd.DataFrame(self.portfolio_values).T
        values_df.index = pd.to_datetime(values_df.index)

        values_df["pct"] = values_df["total"].pct_change().fillna(0)
        values_df["pct_cumprod"] = values_df["pct"].add(1).cumprod() - 1
        values_df["pct_std"] = values_df["pct"].rolling(30).std()

        # orders data
        orders_df = self.get_orders()
        orders_df["value"] = orders_df["quantity"] * orders_df["price"]


        class result:
            excess_return = values_df["pct_cumprod"].iloc[-1]
            return_stddev = values_df["pct"].std()
            days = len(self.portfolio_values)
            annualised_excess_return = ((1 + excess_return) ** (252 / days)) - 1
            sharpe = annualised_excess_return / (values_df["pct"].std() * np.sqrt(252))
            max_drawdown = ((values_df["total"] / values_df["total"].rolling(252).max()) - 1).min()
            calmar_ratio = annualised_excess_return / abs(max_drawdown)
            orders = len(orders_df)
            var_90 = values_df["pct"].quantile(0.1)
            var_95 = values_df["pct"].quantile(0.05)
            var_99 = values_df["pct"].quantile(0.01)


        return result



    def plot_results(self, show=True, matplot_theme="seaborn-v0_8", volatility_window=30):
        # Make a grid for plots:
        # the main plot (70%w, 70%h) is the portfolio value over time
        # the standard deviation plot (70%w, 30%h)
        # results/metrics table on right (30%w, 100%h)

        # set the theme
        plt.style.use(matplot_theme)

        # make plt figure
        fig = plt.figure(figsize=(10, 10))

        # set title
        fig.suptitle("Backtest Results", fontsize=20)

        # make the axes
        ax1 = plt.subplot2grid((10, 10), (0, 0), rowspan=7, colspan=7)
        ax2 = plt.subplot2grid((10, 10), (7, 0), rowspan=3, colspan=7)
        # ax3 = plt.subplot2grid((10, 10), (0, 7), rowspan=10, colspan=3)


        values_df = pd.DataFrame(self.portfolio_values).T
        values_df.index = pd.to_datetime(values_df.index)

        values_df["pct_change"] = values_df["total"].pct_change().fillna(0)
        values_df["pct_cumprod"] = values_df["total"].pct_change().fillna(0).add(1).cumprod() 

        ### AX1 ###        
        # plot the portfolio value
        ax1.plot(values_df.index, values_df["pct_cumprod"] - 1, label="Portfolio Returns")

        # plot zero line
        ax1.plot(values_df.index, [0]*len(values_df), color="#cccccc", alpha=0.3, linestyle="--")

        # Underlying assets
        # plot the mean and standard deviation of the underlying assets
        mean = self.prices.pct_change().add(1).cumprod().mean(axis=1)
        std = self.prices.pct_change().add(1).cumprod().std(axis=1)

        ax1.plot(mean.index, mean - 1, color="green", alpha=0.5, label="Underlying Assets")
        ax1.fill_between(mean.index, mean - std - 1, mean + std - 1, color="green", alpha=0.1)
        ax1.legend()


        # plot the volume traded per day as a bar chart on the bottom of ax1
        traded_volumes = [sum([abs(i[1]) for i in self.placed_orders.values() if i[4] == date]) for date in self.portfolio_values.keys()]
        ax1_2 = ax1.twinx()
        ax1_2.set_ylim((0, max(traded_volumes) * 4))
        # set width to fill all space between the days
        ax1_2.bar(self.portfolio_values.keys(), traded_volumes, color="green", label="Traded Volume", width=3)

        # set legend
        ax1.legend(labels=["Portfolio Returns", "Traded Volume"], loc="upper left")

        # set y axis to percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
        ax1_2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}k".format(x/1000)))

        # set y axis labels
        ax1.set_ylabel("Portfolio Returns")
        ax1_2.set_ylabel("Traded Volume")

        ### AX2 ###
        # volatility plot
        volatility = values_df["pct_change"].rolling(volatility_window).mean()

        # plot fill between where negative values are red
        # and positive values are green
        ax2.fill_between(volatility.index, volatility, color="red", alpha=0.5, where=volatility < 0)
        ax2.fill_between(volatility.index, volatility, color="green", alpha=0.5, where=volatility >= 0)

        # set y limits so mean is in the middle
        ax2.set_ylim((-volatility.abs().max()*1.1, volatility.abs().max()*1.1))


        ### AX3 ###
        # results = self.get_results()

        # tables = {
        #     "Excess Return": "{:.2%}".format(results.excess_return)
        #     , "Annualised Excess Return": "{:.2%}".format(results.annualised_excess_return)
        #     , "Sharpe": "{:.2f}".format(results.sharpe)
        #     , "Days": "{:.0f}".format(results.days)
        # }

        # ax3.axis("off")
        # fontsize = 18
        # ax3.table(cellText=[[i, tables[i]] for i in tables], colLabels=["Metric", "Value"], loc="center", fontsize=fontsize)


        fig.tight_layout()
        
        # show the plot if specified
        if show:
            plt.show()

        return fig



    def _get_portfolio_value_(self, date):
        """
        Get the portfolio value
        """

        # 1 - Get the total value of the portfolio
        total_value = self.cash + (self.prices.loc[date] * self.positions).sum()

        return total_value
        


    def _place_order_(self, symbol, quantity, price, date):
        """
        Place an order

        ### Parameters:
        - `symbol`: The asset to trade
        - `quantity`: The quantity to trade
        - `price`: The price to trade
        - `date`: The date of the trade

        ### Returns:
        `None`
        """


        self._log_(f"Placing {'BUY' if quantity > 0 else 'SELL'} order for {quantity:,.0f} '{symbol}' at {price:,.3f} on {date}")

        # 1 - Check the parameters are correct type
        if not isinstance(symbol, str):
            print("Symbol:", symbol)
            raise TypeError("symbol must be a string")
        if not isinstance(quantity, (int, float, np.int64, np.float64)):
            print("Quantity:", quantity)
            print(type(quantity))
            raise TypeError("quantity must be an integer or float")
        if not isinstance(price, (int, float)):
            print("Price:", price)
            raise TypeError("price must be an integer or float")
        if not isinstance(date, pd.Timestamp):
            print("Date:", date)
            raise TypeError("date must be a pandas Timestamp")
        
        # 2 - Get the new quantity
        old_quantity = self.positions.get(symbol, 0)
        new_quantity = old_quantity + quantity

        # 3 - Check the new quantity is valid
        if new_quantity < 0 and not self.allow_shorting:
            raise ValueError("quantity cannot be negative")
    
        # 4 - Calculate the change in cash after applying fees
        # 4.1 - Calculate the value of the trade
        trade_value = quantity * price
        
        # 4.2 - Calculate the fees
        # It should always be positive, since we are always deducting
        fees = abs(trade_value * (self.commission + self.slippage)) 

        # 4.3 - Calculate the change in cash
        change_in_cash = -trade_value - fees

        # 5 - Check the change in cash is valid
        if self.cash + change_in_cash < 0:
            raise ValueError("not enough cash to place order")
        
        # 6 - Update the cash and positions
        self.cash += change_in_cash

        # 7 - Update the orders
        self.order_id += 1
        self.placed_orders[self.order_id] = [symbol, quantity, price, "BUY" if quantity > 0 else "SELL", date]

        # 8 - Update the positions
        self.positions[symbol] = new_quantity



            

        
    def _close_position_(self, symbol, price, date):
        """
        Close a position
        """

        current_quantity = self.positions.get(symbol, 0)
        self._place_order_(symbol, -current_quantity, price, date)



    def _get_close_quantity_(self, current_weight, previous_weights, current_prices):
        """
        Get the requried quantity to sell the required assets
        """

        # 1 - Get the stocks which had a previous weight, and now is 0
        #     - These are the stocks to sell
        stocks_to_sell = previous_weights[previous_weights != 0].index.difference(current_weight[current_weight != 0].index)

        return stocks_to_sell

        
    def _get_expansion_quantity_(self, current_weight, previous_weights, current_prices):
        """
        Get the the assets which require a change in quantity
        """

        # 1 - Get the change in weights
        change_in_weights = (current_weight - previous_weights)

        # 2 - Remove the assets which are to be closed, these have already been closed
        stocks_to_sell = previous_weights[previous_weights != 0].index.difference(current_weight[current_weight != 0].index)
        change_in_weights = change_in_weights.drop(stocks_to_sell)

        # 3 - Assign value per asset
        assigned_values = (change_in_weights * (self.cash * (1 - self.commission - self.slippage)))
        assigned_quantities = assigned_values // current_prices

        return assigned_quantities
        



    def _get_required_change_in_quantity_(self, current_weights, previous_weights, current_prices):
        """
        Get the required change in quantity
        """

        # print(f"Value: {self.portfolio_value:,.0f}, Cash: {self.cash:,.0f}")

        # 1 - Get the required change in quantity
        change_in_weights = (current_weights - previous_weights)
        assigned_values = change_in_weights * self.portfolio_value

        # pprint(current_weights[current_weights != 0].to_dict())
        # pprint(previous_weights[previous_weights != 0].to_dict())
        # pprint(change_in_weights[change_in_weights != 0].to_dict())
        # pprint(assigned_values[assigned_values != 0].to_dict())

        # calculate the amount of money that will be used to trade
        cash_to_trade = assigned_values.abs().sum()
        # print(f"Cash to trade (gross): {cash_to_trade:,.0f}")

        # discount the amount of money by the commission and slippage
        cash_to_trade = cash_to_trade / (1 + self.commission + self.slippage)
        # print(f"Cash to trade (net): {cash_to_trade:,.0f}")

        sell_value = assigned_values[assigned_values < 0].sum()
        buy_value = assigned_values[assigned_values > 0].sum()

        # print(f"Buy value: {buy_value:,.0f}, Sell value: {sell_value:,.0f}")

        # print("-"*40)

        # use this amount of money to calculate the change in quantity
        required_change_in_quantity = (cash_to_trade * change_in_weights) / current_prices
        
        # any quantity has to be an integer, so round down
        required_change_in_quantity = required_change_in_quantity.apply(lambda x: math.floor(x) if not np.isnan(x) else x)


        # 2 - Check the weights are valid        
        for i in required_change_in_quantity:
            if np.isnan(i):
                raise ValueError("weights must be present for all assets")
            
        return required_change_in_quantity
                


    def _get_next_rebalance_date_(self, date, rebalance_frequency):
        """
        Get the next rebalance date
        """

        if rebalance_frequency == "Y":
            return date + pd.offsets.DateOffset(years=1)
        elif rebalance_frequency == "H":
            return date + pd.offsets.DateOffset(months=6)
        elif rebalance_frequency == "Q":
            return date + pd.offsets.QuarterEnd(0)
        elif rebalance_frequency == "M":
            return date + pd.offsets.MonthEnd(0)
        elif rebalance_frequency == "W":
            return date + pd.offsets.Week(0)
        elif rebalance_frequency == "D":
            return date + pd.offsets.Day(0)
        elif rebalance_frequency == False:
            return None
        else:
            raise ValueError("rebalance_frequency must be either 'M', 'W', or 'D'")


    def _log_(self, message):
        """
        Log a message if verbose is True
        """

        if self.verbose:
            print(message)


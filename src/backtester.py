import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, data: pd.DataFrame, config: dict, logger=None):
        """
        data: MultiIndex DataFrame with index (date, symbol) and columns ['open', 'high', 'low', 'close', 'volume']
        config: dictionary loaded from config.toml
        """
        self.logger = logger
        self.data = data.sort_index()
        self.config = config['Backtest']
        self.dates = sorted(set(data.index.get_level_values(0)))

        self.cash = config.get('initial_cash', 1e6)
        self.interest_rate = config.get('interest_rate', 0.0)  # annual rate
        self.transaction_cost = config.get('transaction_cost', 0.0)  # per share
        self.margin_requirement = config.get('margin_requirement', 1.5)  # multiplier

        self.portfolio = {}  # symbol -> quantity
        self.portfolio_history = []  # daily snapshots
        self.cash_history = []
        self.order_history = []

    def run(self, agent, log=False):
        '''
        :param agent: The agent class should have the act function, which takes the date, data of that day, current portfolio, and current cash,
        and return a list of orders of the form (symbol: str, limit_price: float, quantity: int)
        :param log: whether to log results or not. The default is False.
        '''
        for i in range(1, len(self.dates)):  # start from second date
            today = self.dates[i]
            yesterday = self.dates[i - 1]

            today_data = self.data.loc[today]
            yesterday_data = self.data.loc[yesterday]

            # Provide only yesterday's market data to the agent
            orders = agent.act(yesterday, yesterday_data, self.portfolio.copy(), self.cash)
            if self.logger and log:
                self.logger.info(f"[{today}] Executing {len(orders)} orders")
            self.execute_orders(today, today_data, orders)
            self.mark_to_market(today, today_data)

    def execute_orders(self, date, day_data, orders):
        executed_orders = []
        for symbol, limit_price, quantity in orders:
            if symbol not in day_data.index:
                continue
            high = day_data.loc[symbol, 'high']
            low = day_data.loc[symbol, 'low']

            price = None
            if quantity > 0 and limit_price >= low:
                price = limit_price
            elif quantity < 0 and limit_price <= high:
                price = limit_price

            if price is not None:
                cost = price * abs(quantity)
                fee = self.get_transaction_cost(price, quantity)

                if quantity > 0 and self.cash >= cost + fee:
                    self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                    self.cash -= (cost + fee)
                    executed_orders.append((symbol, price, quantity))
                elif quantity < 0:
                    # For shorts: apply margin constraint
                    total_short_value = sum(
                        max(0, -q) * day_data.loc[sym, 'close']
                        for sym, q in self.portfolio.items()
                    )
                    available_margin = self.cash * self.margin_requirement

                    if total_short_value + cost <= available_margin:
                        self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                        self.cash += (cost - fee)
                        executed_orders.append((symbol, price, quantity))

        self.order_history.append((date, executed_orders))

    def mark_to_market(self, date, day_data):
        portfolio_value = self.cash
        for symbol, quantity in self.portfolio.items():
            if symbol in day_data.index:
                price = day_data.loc[symbol, 'close']
                portfolio_value += quantity * price
        self.portfolio_history.append(portfolio_value)
        self.cash_history.append(self.cash)

    def get_transaction_cost(self, price, quantity):
        return abs(quantity) * self.transaction_cost

    def evaluate(self):
        dates = self.dates[1:]
        index = pd.to_datetime(dates)

        df = pd.DataFrame({
            'portfolio_value': self.portfolio_history,
            'cash': self.cash_history
        }, index=index)

        # Log returns for portfolio
        log_returns = np.log(df['portfolio_value'] / df['portfolio_value'].shift(1)).dropna()
        total_return = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1
        annualized_return = log_returns.mean() * 252
        volatility = log_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - self.interest_rate) / volatility if volatility > 0 else np.nan

        # Drawdown
        drawdown = df['portfolio_value'] / df['portfolio_value'].cummax() - 1
        max_drawdown = drawdown.min()

        # Benchmark (SPY) log returns
        spy_prices = []
        for date in dates:
            try:
                spy_prices.append(self.data.loc[date, "SPY"]["close"])
            except KeyError:
                spy_prices.append(float('nan'))

        spy_series = pd.Series(spy_prices, index=index).ffill()
        spy_log_returns = np.log(spy_series / spy_series.shift(1)).dropna()

        # Align returns
        aligned = log_returns.align(spy_log_returns, join='inner')
        aligned_portfolio, aligned_spy = aligned

        if len(aligned_portfolio) > 1:
            beta = aligned_portfolio.cov(aligned_spy) / aligned_spy.var()
            alpha = (aligned_portfolio.mean() - beta * aligned_spy.mean()) * 252  # annualized
            excess_return = (aligned_portfolio.mean() - aligned_spy.mean()) * 252
        else:
            beta = alpha = excess_return = float('nan')

        return {
            'total_log_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'excess_return': excess_return,
            'beta': beta,
            'alpha': alpha
        }

    def plot(self, agent_name=""):
        """
        Plot portfolio value, cash, drawdown, and SPY benchmark performance.
        """
        # Handle alignment of histories
        dates = self.dates[1:] if len(self.portfolio_history) == len(self.dates) - 1 else self.dates
        index = pd.to_datetime(dates)

        df = pd.DataFrame({
            "Portfolio Value": self.portfolio_history,
            "Cash": self.cash_history
        }, index=index)

        # Compute drawdown
        df["Drawdown"] = df["Portfolio Value"] / df["Portfolio Value"].cummax() - 1

        # Benchmark: SPY total return assuming $1 initial investment
        spy_prices = []
        for date in dates:
            try:
                spy_prices.append(self.data.loc[date, "SPY"]["close"])
            except KeyError:
                spy_prices.append(float('nan'))  # in case SPY is missing on a day

        spy_series = pd.Series(spy_prices, index=index).ffill()
        spy_normalized = spy_series / spy_series.iloc[0] * df["Portfolio Value"].iloc[0]
        df["SPY Benchmark"] = spy_normalized

        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(df.index, df["Portfolio Value"], label="Portfolio Value", color="blue", linewidth=2)
        ax1.plot(df.index, df["SPY Benchmark"], label="SPY Benchmark", color="black", linestyle="--", alpha=0.8)
        ax1.plot(df.index, df["Cash"], label="Cash", color="green", linestyle="--", alpha=0.6)
        ax1.set_ylabel("Value ($)")
        ax1.set_title(f"Portfolio Performance of {agent_name} vs. SPY Benchmark")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Drawdown on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["Drawdown"], label="Drawdown", color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown", color="red")
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.show()

    def get_portfolio_value(self, date):
        if date not in self.dates:
            raise ValueError("Invalid date")
        idx = self.dates.index(date)
        return self.portfolio_history[idx]

    def reset(self):
        """
        Reset the simulation state so that the backtester can be reused with another agent.
        """
        self.cash = self.config.get('initial_cash', 1e6)
        self.portfolio = {}
        self.portfolio_history = []
        self.cash_history = []
        self.order_history = []

        if self.logger:
            self.logger.info("Backtester state has been reset.")
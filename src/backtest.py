import pandas as pd
import numpy as np

class Backtest:
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

    def run(self, agent):
        '''
        :param agent: The agent class should have the act function, which takes the current data, current day data, current portfolio, and current cash,
        and return a list of orders
        '''
        for i in range(1, len(self.dates)):  # start from second date
            today = self.dates[i]
            yesterday = self.dates[i - 1]

            today_data = self.data.loc[today]
            yesterday_data = self.data.loc[yesterday]

            # Provide only yesterday's market data to the agent
            orders = agent.act(yesterday, yesterday_data, self.portfolio.copy(), self.cash)
            if self.logger:
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
        df = pd.DataFrame({
            'portfolio_value': self.portfolio_history,
            'cash': self.cash_history
        }, index=self.dates)

        returns = df['portfolio_value'].pct_change().dropna()
        total_return = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annualized_return - self.interest_rate) / volatility if volatility > 0 else np.nan

        drawdown = df['portfolio_value'] / df['portfolio_value'].cummax() - 1
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

    def get_portfolio_value(self, date):
        if date not in self.dates:
            raise ValueError("Invalid date")
        idx = self.dates.index(date)
        return self.portfolio_history[idx]

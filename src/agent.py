from abc import ABC, abstractmethod
import pandas as pd

class Agent(ABC):
    @abstractmethod
    def act(self, date: pd.Timestamp, market_data: pd.DataFrame, portfolio: dict, cash: float):
        """
        Decide what orders to place based on past market data.

        Parameters:
            date: date of the last available trading day (used for context/logging)
            market_data: DataFrame for that date, indexed by symbol with columns ['open', 'high', 'low', 'close', 'volume']
            portfolio: current holdings {symbol: quantity}
            cash: current available cash

        Returns:
            list of (symbol: str, limit_price: float, quantity: int)
        """
        pass


# This is the RLAgent to be trained later
class RLAgent(Agent):
    def __init__(self, state_builder, policy_model, config=None):
        """
        state_builder: callable that transforms raw market data into agent state
        policy_model: callable that maps state -> action(s)
        config: additional agent-specific parameters (like window size, max position, etc.)
        """
        self.state_builder = state_builder
        self.policy_model = policy_model
        self.config = config or {}
        self.history = {}  # symbol -> list of (date, close)

    def act(self, date, market_data, portfolio, cash):
        # Update historical prices
        for symbol in market_data.index:
            close = market_data.loc[symbol, 'close']
            self.history.setdefault(symbol, []).append((date, close))

        # Step 1: Build agent's current state
        state = self.state_builder(self.history, portfolio, cash)

        # Step 2: Get actions from policy model
        actions = self.policy_model(state)

        # Step 3: Translate model outputs into order tuples
        return self._format_actions(actions, market_data)

    def _format_actions(self, actions, market_data):
        """
        Convert internal model actions into (symbol, limit_price, quantity) tuples.
        You may want to clamp quantities or round prices here.
        """
        formatted = []
        for action in actions:
            symbol = action['symbol']
            price = action['limit_price']
            qty = int(action['quantity'])
            if symbol in market_data.index:
                formatted.append((symbol, price, qty))
        return formatted
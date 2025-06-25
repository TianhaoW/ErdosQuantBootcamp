import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, q=0.0, option_type='call'):
        """
        Parameters:
            S : float — Current stock price
            K : float — Strike price
            T : float — Time to maturity (in years)
            r : float — Annual risk-free interest rate
            sigma : float — Annual volatility of the underlying
            q : float — Annual continuous dividend yield
            option_type : 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()

        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        self._compute_d1_d2()

    def _compute_d1_d2(self):
        if self.T <= 0 or self.sigma <= 0:
            self.d1 = self.d2 = np.nan
        else:
            numerator = np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
            denominator = self.sigma * np.sqrt(self.T)
            self.d1 = numerator / denominator
            self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.option_type == 'call':
            return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)

    def delta(self):
        if self.option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:
            return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)

    def gamma(self):
        return np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self):
        term1 = -self.S * norm.pdf(self.d1) * self.sigma * np.exp(-self.q * self.T) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return term1 - term2 - term3
        else:
            term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return term1 + term2 + term3

    def rho(self):
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

    def summary(self):
        return {
            'price': self.price(),
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }

    def set_params(self, **kwargs):
        for key in ['S', 'K', 'T', 'r', 'sigma', 'q', 'option_type']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        self.option_type = self.option_type.lower()
        self._compute_d1_d2()

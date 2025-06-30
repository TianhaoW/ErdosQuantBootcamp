# ErdosQuantBootcamp

This repository contains my projects for the **Erdos Institute Quantitative Finance Bootcamp**, Summer 2025.

It includes four mini-projects exploring key topics in quantitative finance, ranging from portfolio optimization to volatility modeling.

---

## Contents

### Mini-Project 1: Portfolio Optimization
- Implement **Markowitz Portfolio Theory**:
  - Compute maximum Sharpe ratio and minimum volatility portfolios.
- Explore **LLM-based portfolio selection**.
- Build a simple **backtesting system** for trading agents. 

### Mini-Project 2: Non-Normal Returns
- Investigate the failure of the **Black-Scholes assumption** that log returns follow a normal distribution in real markets.
- Discuss the causes and implications of **fat-tailed distributions** in financial returns.

### Mini-Project 3: Option Pricing
- Revisit the **Black-Scholes PDE** as an inverse heat equation.
- Calculate and interpret option **Greeks**.

### Mini-Project 4: Modeling Volatility
- Compare the **Black-Scholes constant volatility assumption** with real market behavior.
- Study the **implied volatility (IV) smile** and relation with the fat-tailed distributions.
- Implement the **GARCH model** for conditional volatility.
- Explore the **Heston stochastic volatility model**.

---

##  User Guide

### 🔑 API Keys
Some parts of this project require access to the OpenAI API.  
Please create a `.env` file in the root directory and add your API key:

```bash
OPENAI_API_KEY=your_actual_openai_api_key

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
stock_data.to_csv("AppleStockData.csv")

stock_data['Price Change'] = stock_data['Adj Close'].diff()
stock_data['Log Return'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
price_changes = stock_data['Price Change'].dropna()
log_returns = stock_data['Log Return'].dropna()


mu, std = norm.fit(log_returns)
fig, axs = plt.subplots(3, 1, figsize=(10, 7))

# Stock price over time
axs[0].plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close Price', color='blue')
axs[0].set_title('AAPL Stock Price Over Time (2020-01-01 to 2023-01-01)')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Adjusted Close Price')
axs[0].legend()

# Change in stock price over time
axs[1].plot(stock_data.index, stock_data['Price Change'], label='Price Change', color='green')
axs[1].set_title('Change in Stock Price Over Time')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price Change')
axs[1].legend()

# Plot the distribution of log returns
sns.histplot(log_returns, kde=True, stat="density", ax=axs[2], label='Log Return', color='orange')

# Plot the PDF of the fitted normal distribution
x = np.linspace(log_returns.min(), log_returns.max(), 100)
p = norm.pdf(x, mu, std)
axs[2].plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
axs[2].set_title('Distribution of Log Returns with Fitted Normal Distribution')
axs[2].set_xlabel('Log Return')
axs[2].set_ylabel('Density')
axs[2].legend()

plt.tight_layout()
plt.show()
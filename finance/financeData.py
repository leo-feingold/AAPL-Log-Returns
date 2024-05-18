import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


stock_ticker = 'AMZN'
start_date = '2020-01-01'
end_date = '2024-05-16'
stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
stock_data.to_csv(F"{stock_ticker}StockData.csv")

stock_data['Price Change'] = stock_data['Adj Close'].diff()
stock_data['Log Return'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
price_changes = stock_data['Price Change'].dropna()
log_returns = stock_data['Log Return'].dropna()


mu, std = norm.fit(log_returns)
val = 0
fig, axs = plt.subplots(val+3, 1, figsize=(10, 7.5))

# Stock price over time
axs[0].plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close Price', color='blue')
axs[0].set_title(f'{stock_ticker} Stock Price Over Time ({start_date} to {end_date})')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Adjusted Close Price')
axs[0].legend()


#sns.histplot(stock_data['Adj Close'],  kde=True, stat="density", ax=axs[val], label='Stock Price', color='purple')

# Change in stock price over time
axs[val+1].plot(stock_data.index, stock_data['Price Change'], label='Price Change', color='green')
axs[val+1].set_title('Change in Stock Price Over Time')
axs[val+1].set_xlabel('Date')
axs[val+1].set_ylabel('Price Change')
axs[val+1].legend()

# Plot the distribution of log returns
sns.histplot(log_returns, kde=True, stat="density", ax=axs[val+2], label='Log Return', color='orange')

# Plot the PDF of the fitted normal distribution
x = np.linspace(log_returns.min(), log_returns.max(), 100)
p = norm.pdf(x, mu, std)
axs[val+2].plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
axs[val+2].set_title('Distribution of Log Returns with Fitted Normal Distribution')
axs[val+2].set_xlabel('Log Return')
axs[val+2].set_ylabel('Density')
axs[val+2].legend()

plt.tight_layout()
plt.show()
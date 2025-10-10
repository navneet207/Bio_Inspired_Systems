import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the objective function to be optimized (profit based on moving averages)
def fitness_function(params, stock_data):
    short_window = int(params[0])  # Short-term moving average
    long_window = int(params[1])   # Long-term moving average

    # Calculate short and long moving averages
    stock_data['short_ma'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
    stock_data['long_ma'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Generate signals: 1 for Buy, -1 for Sell, 0 for Hold
    stock_data['signal'] = 0
    stock_data['signal'][short_window:] = np.where(stock_data['short_ma'][short_window:] > stock_data['long_ma'][short_window:], 1, -1)

    # Calculate returns
    stock_data['daily_returns'] = stock_data['Close'].pct_change()
    stock_data['strategy_returns'] = stock_data['daily_returns'] * stock_data['signal'].shift(1)

    # Calculate the total profit
    total_profit = stock_data['strategy_returns'].sum()
    
    return -total_profit  # Since we are minimizing, return the negative of profit

# Grey Wolf Optimizer (GWO) implementation
class GreyWolfOptimizer:
    def __init__(self, fitness_function, dim, stock_data, num_wolves=30, max_iter=25):
        self.fitness_function = fitness_function
        self.dim = dim
        self.stock_data = stock_data
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha = None
        self.beta = None
        self.delta = None
        self.positions = np.random.uniform(5, 50, (self.num_wolves, self.dim))  # Random positions for window sizes
        self.fitness = np.array([self.fitness_function(pos, self.stock_data) for pos in self.positions])

    def update_position(self, alpha_pos, wolf_pos, A, C):
        D = np.abs(C * alpha_pos - wolf_pos)
        return alpha_pos - A * D

    def optimize(self):
        for t in range(self.max_iter):
            sorted_indices = np.argsort(self.fitness)
            self.alpha = self.positions[sorted_indices[0]]
            self.beta = self.positions[sorted_indices[1]]
            self.delta = self.positions[sorted_indices[2]]

            A = 2 * np.random.random(self.positions.shape) - 1
            C = 2 * np.random.random(self.positions.shape)

            for i in range(self.num_wolves):
                if np.random.rand() < 0.5:
                    self.positions[i] = self.update_position(self.alpha, self.positions[i], A[i], C[i])
                else:
                    self.positions[i] = self.update_position(self.beta, self.positions[i], A[i], C[i])

                self.positions[i] = np.clip(self.positions[i], 5, 50)

            self.fitness = np.array([self.fitness_function(pos, self.stock_data) for pos in self.positions])

            print(f"Iteration {t+1}/{self.max_iter}, Best Profit: {-self.fitness.min():.4f}")

        return self.alpha, -self.fitness.min()  # Return the best parameters and profit

# Fetch stock data (for example, Apple's stock data)
symbol = 'AAPL'
stock_data = yf.download(symbol, start="2010-01-01", end="2022-01-01")

# GWO to optimize moving average windows
gwo = GreyWolfOptimizer(fitness_function, dim=2, stock_data=stock_data, num_wolves=30, max_iter=25)
best_params, best_profit = gwo.optimize()

print(f"Best Parameters: Short Window: {best_params[0]}, Long Window: {best_params[1]}")
print(f"Best Profit: {best_profit:.4f}")

# Plot the stock price and the optimized moving averages
short_window_opt = int(best_params[0])
long_window_opt = int(best_params[1])

stock_data['short_ma'] = stock_data['Close'].rolling(window=short_window_opt, min_periods=1).mean()
stock_data['long_ma'] = stock_data['Close'].rolling(window=long_window_opt, min_periods=1).mean()

plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Stock Price')
plt.plot(stock_data['short_ma'], label=f'Short MA ({short_window_opt} days)')
plt.plot(stock_data['long_ma'], label=f'Long MA ({long_window_opt} days)')
plt.title(f'{symbol} Stock Price and Optimized Moving Averages')
plt.legend()
plt.show()

# File that contains all the helper functions used in the main file

from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def compute_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def implied_t_costs(weights_t, weights_t_1, fixed_t_costs, relative_t_costs):
    """
    weights_t: weights at time t
    weights_t_1: weights at time t-1
    relative_t_costs: relative transaction costs, may be a vector or a scalar
    fixed_t_costs: fixed transaction costs, may be a vector or a scalar
    """
    t_costs = np.sum(fixed_t_costs*np.sign(abs(weights_t-weights_t_1))) + np.sum(relative_t_costs*np.abs(weights_t - weights_t_1))
    return t_costs

def generate_index(X): 
    rweights = np.random.uniform(0,1,X.shape[1])
    rweights = rweights/np.sum(rweights)

    portfolio = np.dot(X,rweights)

    return portfolio, rweights 

def regression1(x_train, y_train, x_test, y_test, alpha, plot=False):
    # Create and train the Lasso regression model
    lasso = Lasso(alpha=alpha, positive = True)  # You can adjust the regularization strength (alpha) as needed
    lasso.fit(x_train, y_train)
    w = lasso.coef_
    # Predict on the test data
    Y_pred = lasso.predict(x_test)
    
    if np.any(w < 0):
        print("WARNING: Some elements in 'w' are negative.")

    if plot:
        plt.plot(y_test, label='Y_test')
        plt.plot(Y_pred, label='Y_pred')
        plt.title('Lasso regression ')
        plt.legend()
        plt.show()
    return  w

def compute_rel_drawdown(cumreturns):
    cummax = np.maximum.accumulate(cumreturns)
    drawdown = (cummax-cumreturns)/cummax
    max_rel_drawdown = np.max(drawdown)
    return max_rel_drawdown

def compute_gains_losses(cum_returns):
    return np.maximum(0, cum_returns), np.minimum(0, cum_returns)

def plot_returns_cumreturns(plt, FIG_SIZE, Y_test, Y_pred, cum_returns_real, cum_returns_pred):
    plt.figure(figsize=FIG_SIZE)

    plt.subplot(1, 2, 1)
    plt.plot(Y_test, label='Realized Returns')
    plt.plot(Y_pred, label='Predicted Returns', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.title('Test Set Predicted vs Realized Returns')
    plt.legend()

    # Plot the cumulative returns
    plt.subplot(1, 2, 2)
    plt.plot(cum_returns_real, label='Realized Cumulative Returns')
    plt.plot(cum_returns_pred, label='Predicted Cumulative Returns', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
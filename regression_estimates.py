from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_predictions(x, y, N_predictions, window=0, plot=True, savefig=True, name="", plot_index=None):
    N = len(x)
    predictions = []
    scores = []
    mses = []
    for i in range(N_predictions):
        w = window if (window == 0) else N - N_predictions - window + i
        X_train = x[w:N + i - N_predictions]
        X_test = x[N + i - N_predictions]
        y_train = y[w:N + i - N_predictions]
        y_test = y[N + i - N_predictions]
        reg = LinearRegression().fit(X_train, y_train)
        predictions.append(reg.predict(X_test.reshape(1, -1))[0])
        scores.append(reg.score(X_train, y_train))
        mses.append(mean_squared_error(y_test.reshape(1, -1), reg.predict(X_test.reshape(1, -1))))
        
    if plot:
        plt.scatter(y[-N_predictions:], predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        if savefig:
            plt.savefig(f'predictions_vs_true_{name}.png')
        
        fig = plt.figure(figsize=(14,5))
        plt.scatter(x[-N_predictions:], predictions, s=5, label="Predictions", color='orange')
        if plot_index is not None:
            plt.xticks(x[::500], [date.strftime("%d.%m.%Y") for date in plot_index[::500]])
        lower_bound = np.subtract(predictions, np.sqrt(mses))
        upper_bound = np.add(predictions, np.sqrt(mses))
        plt.fill_between(x[-N_predictions:].flatten(), lower_bound, upper_bound, facecolor='red', alpha=0.2)
        plt.plot(x[-N_predictions:].flatten(), lower_bound, linewidth=0.5, alpha=0.8, color='red')
        plt.plot(x[-N_predictions:].flatten(), upper_bound, linewidth=0.5, alpha=0.8, color='red', label='Mean squared error')
        plt.plot(x, y, linewidth=1, label="Training data")
        fig.autofmt_xdate()
        plt.grid(alpha=0.5)
        plt.ylabel("Normalized prices")
        plt.legend()
        plt.title(f'Generalized model 1 for predicting {name} prices')
        if savefig:
            plt.savefig(f'generalized_model_{name}_prices.png')
            
    return predictions, scores, mses

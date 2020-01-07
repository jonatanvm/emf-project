from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def p_value(y, x):
    X = sm.add_constant(x)
    est = sm.OLS(y, X)
    fit = est.fit()
    ttest = pd.DataFrame({'t': fit.tvalues, 'P>|t|':fit.pvalues})
    ttest.index = [f'x{i}' if i > 0 else 'const' for i in range(len(fit.tvalues)) ]
    ttest
    return ttest, fit.pvalues

def hedge_ratio(v_hat,u_hat):
    return v_hat.T.dot(u_hat)/v_hat.T.dot(v_hat)

def regress(x, y, N_predictions, window=0, plot=True, savefig=True, name="", plot_index=None):
    N = len(x)
    predictions = []
    scores = []
    mses = []
    p_values= []
    true_values = x[-N_predictions:]
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
        # xxx
        #X2 = sm.add_constant(X_train)
        #est = sm.OLS(y_train, X_train)
        #est2 = est.fit()
        #print(est2.summary())
        #print(dir(est2))
        #print(["%#6.3f" % (est2.pvalues[i]) for i in range(N_predictions)])
        _, pvs = p_value(y, x)
        for p in pvs:
            p_values.append(p)
    
    if plot:
        fig = plt.figure(figsize=(6,6))
        plt.scatter(y[-N_predictions:], predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        if savefig:
            plt.savefig(f'images/predictions_vs_true_{name}_window_{window}.png',bbox_inches='tight', pad_inches=0)
        
        fig = plt.figure(figsize=(14,5))
        plt.scatter(true_values, predictions, s=5, label="Predictions", color='orange')
        if plot_index is not None:
            plt.xticks(x[::500], [date.strftime("%d.%m.%Y") for date in plot_index[::500]])
        lower_bound = np.subtract(predictions, np.sqrt(mses))
        upper_bound = np.add(predictions, np.sqrt(mses))
        plt.fill_between(true_values.flatten(), lower_bound, upper_bound, facecolor='red', alpha=0.2)
        plt.plot(true_values.flatten(), lower_bound, linewidth=0.5, alpha=0.8, color='red')
        plt.plot(true_values.flatten(), upper_bound, linewidth=0.5, alpha=0.8, color='red', label='Mean squared error')
        plt.plot(x, y, linewidth=1, label="Training data")
        fig.autofmt_xdate()
        plt.grid(alpha=0.5)
        plt.ylabel("Normalized prices")
        plt.legend()
        plt.title(f'Generalized model 1 for predicting {name} prices')
        if savefig:
            plt.savefig(f'images/generalized_model_{name}_prices_window_{window}.png',bbox_inches='tight', pad_inches=0)
    
    print(f"All p-values > 0.05: {np.sum(np.array(p_values) > 0.05)}")
    return true_values, predictions, scores, mses

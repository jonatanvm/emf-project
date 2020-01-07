import matplotlib.pyplot as plt
import numpy as np

def plot_regression(regression, data, name, savefig=False):
    x = np.array(list(range(data.shape[0]))).reshape(-1,1)
    predictions = regression.predict(x)
    delta = data - regression.predict(x)
    mse = np.mean(delta**2)
    if len(data.shape) == 1 or data.shape[1] == 1:
        fig = plt.figure(figsize=(14,5))
        plt.plot(predictions,linewidth=1.0)
        plt.plot(data,linewidth=1.0)

        lower_bound = np.subtract(predictions, np.sqrt(mse))
        upper_bound = np.add(predictions, np.sqrt(mse))
        plt.fill_between(x.flatten(), lower_bound.flatten(), upper_bound.flatten(), facecolor='red', alpha=0.2)
        plt.plot(x, lower_bound, linewidth=0.5, alpha=0.8, color='red')
        plt.plot(x, upper_bound, linewidth=0.5, alpha=0.8, color='red', label='Mean squared error')
        plt.savefig(f"images/{name}")
    else:
        for i in range(data.shape[1]):
            fig = plt.figure(figsize=(14,5))
            plt.plot(predictions[:,i],linewidth=1.0)
            plt.plot(data[:,i],linewidth=1.0)

            lower_bound = np.subtract(predictions, np.sqrt(mse))
            upper_bound = np.add(predictions, np.sqrt(mse))
            
            plt.fill_between(x.flatten(), lower_bound[:,i], upper_bound[:,i], facecolor='red', alpha=0.2)
            plt.plot(x, lower_bound[:,i], linewidth=0.5, alpha=0.8, color='red')
            plt.plot(x, upper_bound[:,i], linewidth=0.5, alpha=0.8, color='red', label='Mean squared error')
            plt.savefig(f"images/{i}_{name}")
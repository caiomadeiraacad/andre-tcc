import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(series, lags=20, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(series, lags=lags, ax=ax[0], title=f'ACF {title}')
    plot_pacf(series, lags=lags, ax=ax[1], title=f'PACF {title}', method='ywm')
    plt.show()

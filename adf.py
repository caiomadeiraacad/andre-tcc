import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Teste de Dickey-Fuller aumentado (ADF)
def dickey_fuller_test(series, title=''):
    result = adfuller(series)
    print(f'Teste de Dickey-Fuller para {title}')
    print(f'Estatística: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Critérios:')
    for key, value in result[4].items():
        print(f'{key}: {value}')
    print('Conclusão:', 'Estacionária' if result[1] < 0.01 else 'Não Estacionária')
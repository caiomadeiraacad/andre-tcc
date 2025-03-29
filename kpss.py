import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def kpss_test(series, title=""):
    result = kpss(series.dropna(), regression="c", nlags="auto")  # Regressão com constante
    print(f"\n Teste KPSS para {title}")
    print(f"Estatística: {result[0]:.4f}")
    print(f"p-valor: {result[1]:.4f}")
    print("Valores Críticos:")
    for key, value in result[3].items():
        print(f"{key}: {value:.4f}")
    print("Conclusão:", "[x] Não Estacionária" if result[1] < 0.05 else "[ok] Estacionária")
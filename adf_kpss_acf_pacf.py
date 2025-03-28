import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

YEAR_COLUMNS_START = 4

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
    
def kpss_test(series, title=""):
    result = kpss(series.dropna(), regression="c", nlags="auto")  # Regressão com constante
    print(f"\n Teste KPSS para {title}")
    print(f"Estatística: {result[0]:.4f}")
    print(f"p-valor: {result[1]:.4f}")
    print("Valores Críticos:")
    for key, value in result[3].items():
        print(f"{key}: {value:.4f}")
    print("Conclusão:", "[x] Não Estacionária" if result[1] < 0.05 else "[ok] Estacionária")

def plot_acf_pacf(series, lags=20, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(series, lags=lags, ax=ax[0], title=f'ACF {title}')
    plot_pacf(series, lags=lags, ax=ax[1], title=f'PACF {title}', method='ywm')
    plt.show()

def make_plot(data: any, title: str, x_label: str, y_label: str, filename: str = "", save: bool = None):
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Ano")
    plt.ylabel("Emissão (Mt CO₂)")
    plt.grid()
    if (save and filename != ""):
        if (filename != ""):
            plt.savefig(filename + ".png")
    plt.show()

if __name__ == '__main__':

    greenhouse_effect = pd.read_csv("serie_temporal_tcc.csv", delimiter=',', on_bad_lines='warn')

    """
    Como o csv não está muito formatado pra fazer uma série temporal
    preciso corrigir algumas coisas:
    """

    # remove a coluna unnamed (eh lixo)
    greenhouse_effect = greenhouse_effect.loc[:, ~greenhouse_effect.columns.str.contains('^Unnamed')]

    years = greenhouse_effect.columns[4:-1]

    # Filtro apenas o brasil aqui e removo as colunas que n sao os anos
    bra = greenhouse_effect[greenhouse_effect['Country Name'] == 'Brazil'].iloc[:, 4:]

    # O uso do melt: transforma, nesse contexto, os anos (que são colunas)
    # em linhas e os valores em uma única coluna
    bra = bra.melt(var_name="Year", value_name="Emission")

    # converter year pra datetime e orderna
    bra["Year"] = pd.to_datetime(bra["Year"], format="%Y")
    bra = bra.sort_values("Year").set_index("Year")

    # Crio uma série temporal
    bra_ts = pd.Series(bra["Emission"].values, index=bra.index)
    # plotando um grafico simples
    # make_plot(data=bra_ts,
    #           title="Emissão de Gases do Efeito Estufa - Brasil", 
    #           x_label="Ano", 
    #           y_label="Emissão (Mt CO₂)",
    #           filename = "bra_ts", 
    #           save=True)
    # remove os NaNs da serie temporal antes de testar
    bra_ts = bra_ts.dropna()

    #  Testes de Estacionariedade
    print("\n=========================================================================")
    print("Testes de Estacionariedade.")
    print("Dickey-Fuller Aumentado [ADF]\Kwiatkowski-Phillips-Schimidt-Shin [KPSS]")
    print("=========================================================================\n")

    dickey_fuller_test(bra_ts, "Emissões do Brasil")
    kpss_test(bra_ts, "Emissões do Brasil")

    print("==================================\n")
    print("Primneiros valores da serie temporal: ")
    print(bra_ts.head())

    print("\n=========================================================================")
    print("Gráficos de Autocorrelação e Autocorrelação parcial.")
    print("=========================================================================\n")
    plot_acf_pacf(bra_ts, title="Emissões do Brasil")


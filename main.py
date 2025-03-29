from acf_pacf import *
from adf import *
from kpss import *
from plot import *
from pmdarima import auto_arima, arima
from scipy.stats import kstest
from scipy.stats import levene
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

YEAR_COLUMNS_START = 4
SPLIT_YEAR = '2000'
"""
Como o csv não está muito formatado pra fazer uma série temporal
preciso corrigir algumas coisas:
"""

def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d

greenhouse_effect = pd.read_csv("serie_temporal_tcc.csv", delimiter=',', on_bad_lines='warn')

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

"""
Pelos 5segundos de pesquisa, tu usa o ARIMA como modelo estatístico 
pra prever dados futuros em uma serie temporal (usando dos dados legados da série).
AR, I, MA

preciso dispor no output os valores de p, d, q
"""

model = auto_arima(
    bra_ts,
    seasonal=False,
    trace=True,
    suppress_warnings=True,
    stepwise=True
)

if __name__ == '__main__':

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

    # ARIMA
    print("\n=========================================================================")
    print("Modelo Arima")
    print("Modelo ARIMA; Akaike (AIC)")
    print("=========================================================================\n")

    print(model.summary())

    forecast = model.predict(n_periods=10)
    print("prev. p/ os proximos 10 anos:", forecast)

    print(f"Melhores parâmetros encontrados: p={model.order[0]}, d={model.order[1]}, q={model.order[2]}")

    print("\n=========================================================================")
    print("Gráfico da série diferenciada em primeira diferença")
    print("=========================================================================\n")
    verificar_differenciacao(bra_ts, "Emissões efeito estufa BRASIL")

    # Diferenciação para estacionariedade
    bra_ts_diff = bra_ts.diff().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16,4))
    plot_acf(bra_ts_diff, lags=24, ax=axes[0])
    print("AXES[0]>>>>> ", axes)
    plot_pacf(bra_ts_diff, lags=24, ax=axes[1], method='ywm')
    plt.show()
    # Ajuste do modelo ARIMA na série diferenciada (autoarima)
    arima_bra = auto_arima(bra_ts_diff,
                            seasonal=True,
                            m=12,  # Periodicidade da sazonalidade
                            trace=True,
                            stepwise=True)

    # Exibir o resumo do modelo ajustado
    print(arima_bra.summary())
    import statsmodels.api as sm

    print("\n=========================================================================")
    print("Validação e Diagnóstico (Resíduos)")
    print("=========================================================================\n")
    print("Resíduos do modelo")
    waste_arima = arima_bra.resid()
    print(f"Resíduos do modelo: {waste_arima}")
    
    print("\n=========================================================================")
    print("Teste de Ljung-Box para verificar autocorrelação dos resíduos")
    print("=========================================================================\n")
    ljung_box = sm.stats.acorr_ljungbox(waste_arima, lags=[10], return_df=True)
    print(f'Resultado do teste de Ljung-Box:\n{ljung_box}')

    print("\n=========================================================================")
    print("Teste de Normalidade dos Resíduos (Kolmogorov-Smirnov)")
    print("=========================================================================\n")
    ks_stat, p_value = kstest(waste_arima, 'norm', args=(np.mean(waste_arima), np.std(waste_arima)))
    print(f'Teste de Kolmogorov-Smirnov para normalidade: p-valor = {p_value}')
    if p_value > 0.01:
        print("Os resíduos seguem uma distribuição normal.")
    else:
        print("Os resíduos não seguem uma distribuição normal.")

    print("\n=========================================================================")
    print("Teste de Levene (homogeneidade das variâncias)")
    print("=========================================================================\n")

    print("A série temporal foi divida em duas partes: antes e depois de 2000")
    group1 = bra_ts[bra_ts.index < SPLIT_YEAR]
    group2 = bra_ts[bra_ts.index >= SPLIT_YEAR]

    print("GRUPO 01: ", group1)
    print("GRUPO 02: ", group2)

    stat, p_value = levene(group1, group2)
    print(f"Resultado do teste: {stat}, p-valor: {p_value}")
    if p_value > 0.05:
        print("[x] Não há evidência de heterocedasticidade (variâncias homogêneas).")
    else:
        print("[ok] Há evidência de heterocedasticidade (variâncias não homogêneas).")
    print("\n=========================================================================")
    print("Teste do Multiplicador de Lagrange (het_arch)")
    print("=========================================================================\n")

    am = arch_model(waste_arima, vol='ARCH', p=1)
    test_arch = am.fit(disp='off')
    print(test_arch.summary())

    # In[195]: Prever 24 passos à frente na série diferenciada
    n_periods = 24
    previsoes_diff = arima_bra.predict(n_periods=n_periods)
    print(f"Previsões diferenciadas: {previsoes_diff}")

    print("\n=========================================================================")
    print("Teste do Multiplicador de Lagrange (usand statsmodel)")
    print("=========================================================================\n")
    # Teste ARCH
    stat, p_lm, f_stat, p_f = het_arch(waste_arima)

    print("\n=========================================================================")
    print("Teste ARCH (Multiplicador de Lagrange)")
    print("=========================================================================\n")
    print(f"Estatística LM: {stat}, p-valor LM: {p_lm}")
    print(f"Estatística F: {f_stat}, p-valor F: {p_f}")

    if p_lm > 0.05:
        print("Não há evidência de heterocedasticidade condicional (não é ARCH).")
    else:
        print("Há evidência de heterocedasticidade condicional (possível efeito ARCH).")
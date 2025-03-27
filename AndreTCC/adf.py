import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

YEAR_COLUMNS_START = 4

temp_serie = pd.read_csv("serie_temporal_tcc.csv", delimiter=',', on_bad_lines='warn')

country_name = temp_serie['Country Name']

# remove a coluna unnamed (eh lixo)
temp_serie = temp_serie.loc[:, ~temp_serie.columns.str.contains('^Unnamed')]

print(temp_serie.head)

years = temp_serie.columns[4:-1]


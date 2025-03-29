import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

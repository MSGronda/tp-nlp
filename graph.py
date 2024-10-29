import pandas as pd
from matplotlib import pyplot as plt

def graph_distribution(files):
    for filename in files:
        with open(filename, "r") as f:
            print(f"Reading {filename}")
            df = pd.read_csv(f)

            colors = ['orange', 'blue', 'red']
            plt.ylabel("% de tweets")
            (df['sentiment'].value_counts(normalize=True) * 100).plot.bar(color=colors)
            plt.title(f'Distribución de Sentimientos con {filename.removesuffix(".csv").capitalize()}')
            plt.show()


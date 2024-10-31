import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def graph_distribution(files):
    for filename in files:
        with open(filename, "r") as f:
            print(f"Reading {filename}")
            df = pd.read_csv(f)

            plt.ylabel("% de tweets")
            (df['sentiment'].value_counts(normalize=True) * 100).plot.bar(color=[v for v in mcolors.TABLEAU_COLORS.values()])
            plt.title(f'Distribución de Sentimientos con {filename.removesuffix(".csv").capitalize()}')
            plt.xlabel("Sentiment")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


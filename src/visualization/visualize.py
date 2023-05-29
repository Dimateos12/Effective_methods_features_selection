import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from features.filters import filters
import venn
import time

config = load_config("my_configuration.yaml")

def venn_diagram(X, y):
    """
    Plot a Venn diagram for four sets.

    Args:
    X (list or array-like): Data points.
    y (list or array-like): Class labels.

    Returns:
    None: Displays the Venn diagram plot.

    Raises:
    TypeError: If X or y are not list or array-like.
    """
    set1 = set(filters(X, y, "ReliefF"))
    set2 = set(filters(X, y, "Mrmr"))
    set3 = set(filters(X, y, "U-test"))
    set4 = set(filters(X, y, "MDFS"))

    labels = venn.get_labels([set1, set2, set3, set4], fill=['number'])
    fig, ax = venn.venn4(labels, names=["ReliefF", "MRMR", "U-test", "MDFS"])
    plt.show()

    plt.show()
    # plt.savefig('../../reports/figures/venn.png')


def compare_scores():
    df1 = pd.read_csv(config['scores_file_Mrmr'])
    df2 = pd.read_csv(config['scores_file_ReliefF'])
    df3 = pd.read_csv(config['scores_file_U-test'])
    df4 = pd.read_csv(config['scores_file_MDFS'])
    # Dane
    dataframes = [df1.sort_values(by='Number of Features'), df2.sort_values(by='Number of Features'),
                  df3.sort_values(by='Number of Features'), df4.sort_values(by='Number of Features')]
    titles = [df1['Feature Selection Method'].unique()[0], df2['Feature Selection Method'].unique()[0],
              df3['Feature Selection Method'].unique()[0], df4['Feature Selection Method'].unique()[0]]

    # Tworzenie podwykresów
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Iteracja przez DataFrame'y i tworzenie wykresów
    metrics = ['ACC', 'AUC', 'MCC', 'F1']
    colors = ['blue', 'orange', 'green', 'brown']
    algorithms = ['MRR', 'U-test', 'ReliefF', 'MDFS']

    for i, metric in enumerate(metrics):
        row = i // 2  # Wiersz podwykresu
        col = i % 2  # Kolumna podwykresu

        axes[row, col].set_xlabel("Ilość cech")
        axes[row, col].set_ylabel("Wartość")
        axes[row, col].set_title(metric)

        for j, df in enumerate(dataframes):
            x = df['Number of Features']
            y = df[metric]

            axes[row, col].plot(x, y, label=algorithms[j], color=colors[j])

        axes[row, col].legend()
        x_ticks = [5, 10, 20, 40, 60, 80, 100, 120, 140]
        axes[row, col].set_xticks(x_ticks)
        y_ticks = np.arange(0.1, 0.75, 0.05)
        axes[row, col].set_yticks(y_ticks)
        axes[row, col].grid(True)

    # Dopasowanie wykresów w układzie
    plt.tight_layout()

    # Wyświetlenie wykresu
    # plt.show()

    plt.savefig(config['figures_path'] + "compare_scores.png")
    plt.close()


def time_compare(X, y):
    num_of_features = [5, 10, 20, 40, 60, 80, 100, 120, 140]
    filters_names = ["ReliefF", "Mrmr", "U-test", "MDFS"]
    time_taken = {name: [] for name in filters_names}

    for num in num_of_features:
        for name in filters_names:
            start_time = time.time()
            filters(X, y, name, num)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_taken[name].append(elapsed_time)

    for name in filters_names:
        plt.plot(num_of_features, time_taken[name], marker='o', label=name)

    plt.xlabel('Ilość cech')
    plt.ylabel('Czas trwania  (sekundy)')
    plt.title('Czas trwania vs ilość cech   ')
    plt.legend()
    plt.show()

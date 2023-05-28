import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from features.filters import filters

config = load_config("my_configuration.yaml")


def plot_confusion_matrix(y_test, y_pred):
    """
    Plots a confusion matrix for binary classification models.

    Args:
    y_test (array-like): True labels of the test data.
    y_pred (array-like): Predicted labels of the test data.

    Returns:
    None. Displays a plot of the confusion matrix.
    """
    conf_mx = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mx)
    fig.colorbar(cax)
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            ax.annotate(
                str(conf_mx[i][j]),
                xy=(j, i),
                horizontalalignment="center",
                verticalalignment="center",
                color="w",
            )
    plt.show()


def venn_diagram(X, y):
    """
    Plot a Venn diagram for three sets.

    Args:
    set1 (list or array-like): First set of values.
    set2 (list or array-like): Second set of values.
    set3 (list or array-like): Third set of values.

    Returns:
    None: Displays the Venn diagram plot.

    Raises:
    TypeError: If any of the sets are not list or array-like.
    """
    set1 = filters(X, y, "ReliefF")
    set2 = filters(X, y, "Mrmr")
    set3 = filters(X, y, "U-test")

    set1format = set(tuple(set1))
    set2format = set(tuple(set2))
    set3format = set(tuple(set3))

    venn_diagram = venn3([set1format, set2format, set3format], ("ReliefF", "Mrmr", "U-test"))
    plt.show()
    # plt.savefig('../../reports/figures/venn.png')


def compare_scores():
    df1 = pd.read_csv(config['scores_file_Mrmr'])
    df2 = pd.read_csv(config['scores_file_ReliefF'])
    df3 = pd.read_csv(config['scores_file_U-test'])
    # Dane
    dataframes = [df1.sort_values(by='Number of Features'), df2.sort_values(by='Number of Features'),
                  df3.sort_values(by='Number of Features')]
    titles = [df1['Feature Selection Method'].unique()[0], df2['Feature Selection Method'].unique()[0],
              df3['Feature Selection Method'].unique()[0]]

    # Tworzenie podwykresów
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Iteracja przez DataFrame'y i tworzenie wykresów
    metrics = ['ACC', 'AUC', 'MCC', 'F1']
    colors = ['blue', 'orange', 'green']
    algorithms = ['MRR', 'U-test', 'ReliefF']

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


def time_compare():
    df1 = pd.read_csv(config['scores_file_Mrmr'])
    df2 = pd.read_csv(config['scores_file_ReliefF'])
    df3 = pd.read_csv(config['scores_file_U-test'])
    # Dane
    df1_sorted = df1.sort_values('Number of Features')
    df2_sorted = df2.sort_values('Number of Features')
    df3_sorted = df3.sort_values('Number of Features')
    # Dane
    x_mrmr = df1_sorted['Number of Features']
    x_relief = df2_sorted['Number of Features']
    x_utest = df3_sorted['Number of Features']
    y_d1 = df1_sorted['Time']
    y_d2 = df2_sorted['Time']
    y_d3 = df3_sorted['Time']

    # Tworzenie wykresu
    # plt.plot(x_mrmr, y_d1, label='MRMR')
    plt.plot(x_relief, y_d2, label='ReliefF')
    plt.plot(x_utest, y_d3, label='U-test')
    plt.xticks([5, 10, 20, 40, 60, 80, 100, 120, 140])

    # Dodanie etykiet i tytułu
    plt.xlabel("Ilość cech")
    plt.ylabel("Czas (sekundy)")
    plt.title("Porównanie wydajności czasowej algorytmów")

    # Dodanie legendy
    plt.legend()
    plt.grid()
    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()

    plt.savefig(config['figures_path'] + "compare_time.png")
    plt.close()

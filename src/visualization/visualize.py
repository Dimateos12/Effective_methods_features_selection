import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from sklearn.metrics import confusion_matrix


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


def venn_diagram(set1, set2, set3):
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
    set1format = set(tuple(pd.DataFrame(set1)))
    set2format = set(tuple(pd.DataFrame(set2)))
    set3format = set(tuple(pd.DataFrame(set3)))
    venn3([set1format, set2format, set3format], ("Set 1", "set 2", "set3"))
    plt.show()
    plt.savefig('././reports/figures/venn.png')

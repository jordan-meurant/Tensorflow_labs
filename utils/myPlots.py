from matplotlib import pyplot as plt
import seaborn as sns


def plot_matrix(matrix, labels=None, vmin=None, vmax=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    if vmin is not None and vmax is not None:
        ax = sns.heatmap(matrix, vmin=0, vmax=1, annot=True)
    else:
        ax = sns.heatmap(matrix, annot=True)

    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    fig.show()

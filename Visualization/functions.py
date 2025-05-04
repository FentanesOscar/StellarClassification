import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt

def plot_redshift_histograms_by_class(
    df,
    target_col='target',
    redshift_col='redshift',
    bins=30,
    density=True,
    histtype='step',
    alpha=0.7,
    figsize=(12, 4),
    colors=None
):
    """
    Plot individual histograms of redshift for each unique target class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the target and redshift columns.
    target_col : str, default 'target'
        Name of the column with class labels.
    redshift_col : str, default 'redshift'
        Name of the column with redshift values.
    bins : int, default 30
        Number of histogram bins.
    density : bool, default True
        If True, normalize histograms so area = 1.
    histtype : str, default 'step'
        Matplotlib histogram type.
    alpha : float, default 0.7
        Transparency level for histogram lines.
    figsize : tuple, default (12, 4)
        Figure size in inches (width, height).
    colors : list, optional
        List of colors for each class; default uses tab10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.
    axs : numpy.ndarray
        Array of Axes objects (one per class).
    """
    classes = df[target_col].unique()
    n = len(classes)

    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]

    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axs = [axs]

    for ax, cls, color in zip(axs, classes, colors):
        data = df[df[target_col] == cls][redshift_col].dropna()
        ax.hist(
            data,
            bins=bins,
            density=density,
            histtype=histtype,
            alpha=alpha,
            color=color
        )
        ax.set_title(f"{cls}")
        ax.set_xlabel(redshift_col)
        ax.set_ylabel('Density' if density else 'Count')

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_histograms_by_class(
    df,
    target_col,
    class_colors=None,
    bins=30,
    histtype='step',
    alpha=0.7,
    figsize=(10, 10),
    legend_loc='upper right'
):
    """
    Plot overlaid histograms for each feature in df, split by class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing features and the target column.
    target_col : str
        Name of the column in df that contains class labels.
    class_colors : dict, optional
        Mapping from class label to color. If None, default 'tab10' colors are used.
    bins : int, optional
        Number of bins for the histograms (default: 30).
    histtype : str, optional
        Type of histogram to draw (default: 'step').
    alpha : float, optional
        Transparency level for histogram fills (default: 0.7).
    figsize : tuple of (width, height), optional
        Figure size in inches (default: (10, 10)).
    legend_loc : str, optional
        Location string for the legend (default: 'upper right').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axs : numpy.ndarray
        Flattened array of Axes objects.
    """
    features = df.columns.drop(target_col)
    n_features = len(features)

    ncols = int(np.ceil(np.sqrt(n_features)))
    nrows = int(np.ceil(n_features / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    classes = df[target_col].unique()
    if class_colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(classes))]
        class_colors = dict(zip(classes, colors))

    for i, feature in enumerate(features):
        ax = axs[i]
        for cls in classes:
            data = df[df[target_col] == cls][feature]
            ax.hist(
                data,
                bins=bins,
                density = True,
                histtype=histtype,
                alpha=alpha,
                label=str(cls),
                color=class_colors[cls]
            )
        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.legend(loc=legend_loc, fontsize='small')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_correlation_heatmap(
    df,
    columns=None,
    figsize=(10, 10),
    cmap='viridis',
    annot=True,
    fmt=".2f",
    title="Correlation Matrix Heatmap",
    rotation=45
):
    """
    Plot a correlation matrix heatmap for the given DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list-like, optional
        Subset of columns to include. If None, all df.columns are used.
    figsize : tuple, optional
        Figure size in inches (default: (10, 10)).
    cmap : str or matplotlib Colormap, optional
        Colormap for the heatmap (default: 'viridis').
    annot : bool, optional
        Whether to annotate cells with correlation values (default: True).
    fmt : str, optional
        Format string for annotations (default: ".2f").
    title : str, optional
        Title of the plot (default: "Correlation Matrix Heatmap").
    rotation : float, optional
        Rotation angle for x-axis labels (default: 45 degrees).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.
    ax : matplotlib.axes.Axes
        The created Axes object.
    """
    if columns is None:
        columns = df.columns
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, interpolation='nearest', cmap=cmap)

    ticks = np.arange(len(columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")

    if annot:
        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(j, i, f"{corr.iat[i, j]:{fmt}}", ha="center", va="center")

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return fig, ax




def plot_boxplots_by_class(
    df, 
    target_col,
    class_order=None,
    figsize=(15, 15),
    patch_artist=True,
    **boxplot_kwargs
):
    """
    Plot boxplots for each feature in df, split by class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing features and the target column.
    target_col : str
        Name of the column in df that contains class labels.
    class_order : list-like, optional
        Specific order of class labels. If None, uses unique values in target_col.
    figsize : tuple, optional
        Figure size in inches (default: (15, 15)).
    patch_artist : bool, optional
        Whether to apply patch_artist to boxplot (default: True to allow facecolors).
    **boxplot_kwargs : dict
        Other keyword args to pass to ax.boxplot, e.g. notch=True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.
    axs : numpy.ndarray
        Flattened array of Axes objects.
    """
    features = df.columns.drop(target_col)
    n_features = len(features)

    ncols = int(np.ceil(np.sqrt(n_features)))
    nrows = int(np.ceil(n_features / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    classes = class_order if class_order is not None else df[target_col].unique()
    labels = [str(c) for c in classes]

    for i, feature in enumerate(features):
        ax = axs[i]
        data_to_plot = [
            pd.to_numeric(df[df[target_col] == c][feature].dropna(), errors='coerce').values
            for c in classes
        ]
        ax.boxplot(data_to_plot, patch_artist=patch_artist, tick_labels=labels, **boxplot_kwargs)
        ax.set_xlabel(feature)
        ax.set_ylabel("Value")
        ax.set_title(f"Boxplot for {feature} across classes")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    return fig, axs

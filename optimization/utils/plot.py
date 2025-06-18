"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import cm
from IPython.display import HTML

def plot_contour(func, ranges, increments, points=None, contour_levels=30, contour_cmap=plt.cm.viridis,
                 special_points=None, title="", show_axis=True, ax=None,
                 points_label="Points", special_points_label="Special Points"):
    """
    Plot a contour of a function over a grid and overlay given points.

    Arguments:
        func: Function to be evaluated over the grid. Must accept array of shape (N, 2).
        ranges: Ranges for x and y as [[xmin, xmax], [ymin, ymax]].
        increments: Array-like of [x_increment, y_increment] or a single value.
        points: Points to overlay, shape (N, 2).
        contour_levels: Number of contour levels.
        contour_cmap: Color map for the contour.
        special_points: Extra points to highlight in red.
        title: Title of the plot.
        show_axis: Whether to show axis.
        ax: Optional matplotlib Axes object.
        points_label: Label for the points to show in the legend.
        special_points_label: Label for the special points to show in the legend.

    Returns:
        fig, ax, c: Figure, Axes, and QuadContourSet objects.
    """

    if np.array(increments).size == 1:
        increments = np.array([increments, increments])

    x = np.arange(ranges[0][0], ranges[0][1], increments[0])
    y = np.arange(ranges[1][0], ranges[1][1], increments[1])
    X, Y = np.meshgrid(x, y)
    Z = func(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(X.shape)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    c = ax.contourf(X, Y, Z, levels=contour_levels, cmap=contour_cmap)
    fig.colorbar(c, ax=ax)

    handles = []
    labels = []

    if points is not None:
        points_plot, = ax.plot(np.array(points)[:, 0], np.array(points)[:, 1], "bo", label=points_label)
        handles.append(points_plot)
        labels.append(points_label)

    if special_points is not None:
        special_plot, = ax.plot(np.array(special_points)[:, 0], np.array(special_points)[:, 1], "rx", label=special_points_label)
        handles.append(special_plot)
        labels.append(special_points_label)

    if handles:
        ax.legend(handles=handles, labels=labels)

    ax.set_title(title)
    if not show_axis:
        ax.axis("off")

    return fig, ax, c


def plot_contour_gif(func, ranges, increments, population_history,
                     contour_levels=30, contour_cmap=cm.viridis,
                     special_points=None, title="Optimization Progress",
                     points_label="Population", special_points_label="Optimum"):
    """
    Plot an animated contour of a function showing population evolution.

    Arguments:
        func: Callable accepting array (N, 2), returning array (N,)
        ranges: [[xmin, xmax], [ymin, ymax]]
        increments: scalar or [x_increment, y_increment]
        population_history: ndarray of shape (G, N, 2)
        contour_levels: number of contour levels
        contour_cmap: matplotlib colormap
        special_points: list of [x, y] global optima (optional)
        title: base title string for the plot
        points_label: label for population points
        special_points_label: label for special points

    Returns:
        IPython.display.HTML animation object
    """
    ## Validate input
    assert population_history.ndim == 3 and population_history.shape[2] == 2, "Expected shape (G, N, 2)"

    ## Ensure increments is [x_inc, y_inc]
    if np.isscalar(increments):
        increments = [increments, increments]

    ## Prepare grid
    x = np.arange(ranges[0][0], ranges[0][1], increments[0])
    y = np.arange(ranges[1][0], ranges[1][1], increments[1])
    X, Y = np.meshgrid(x, y)
    Z = func(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(X.shape)

    ## Set up figure and base contour
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, Z, levels=contour_levels, cmap=contour_cmap)
    fig.colorbar(c, ax=ax)
    scat = ax.plot([], [], 'bo', label=points_label)[0]

    if special_points is not None:
        special_plot = ax.plot(np.array(special_points)[:, 0], np.array(special_points)[:, 1], 'rx', label=special_points_label)[0]
    else:
        special_plot = None

    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    ax.legend()
    ax.set_title(title)

    ## Animation function
    def update(frame):
        pop = population_history[frame]
        scat.set_data(pop[:, 0], pop[:, 1])
        ax.set_title(f"{title} (Generation {frame + 1}/{population_history.shape[0]})")
        return scat,

    anim = animation.FuncAnimation(fig, update, frames=population_history.shape[0],
                                   interval=200, blit=True)

    plt.close(fig)  # Avoid static display in notebooks
    return HTML(anim.to_jshtml())
import numpy as np


def bound_box(points):
    """Compute the bound box of a point cloud.

    Args:
        points: A (-1, 3) np array containing the points

    Returns:
        x_min, x_max, y_min, y_max, z_min, z_max
    """
    min_x = np.amin(points[:, 0])
    min_y = np.amin(points[:, 1])
    min_z = np.amin(points[:, 2])

    max_x = np.amax(points[:, 0])
    max_y = np.amax(points[:, 1])
    max_z = np.amax(points[:, 2])

    return min_x, max_x, min_y, max_y, min_z, max_z


def grid_centers(bound_box, cell_size):
    """Compute the centers of a grid.

    Args:
        bound_box: a list containing (x_min, x_max, y_min, y_max, z_min, z_max)
        cell_size: the size of cell

    Returns:
        A (-1, 3) shape np array containing the grid centers
    """
    grid = np.mgrid[
        bound_box[0] + cell_size / 2: bound_box[1]: cell_size,
        bound_box[2] + cell_size / 2: bound_box[3]: cell_size,
        bound_box[4] + cell_size / 2: bound_box[5]: cell_size,
    ].T

    grid = np.reshape(grid, (-1, 3))

    return grid

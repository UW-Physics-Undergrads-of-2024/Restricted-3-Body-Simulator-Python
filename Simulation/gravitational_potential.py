from logging import DEBUG, getLogger, info, StreamHandler
from matplotlib.pyplot import figure, contour, colorbar, show
from numba import jit
from numpy import meshgrid, linspace, sqrt, array
from sys import stdout
from time import process_time, perf_counter


@jit(nopython=True)
def gravity_potential(m1: float, m2: float, m2_orbital_radius: float,
                      x_points: array, y_points: array) -> array:
    """
    Creates a contour plot of the gravitational potential for the smaller third mass. The proportion of M1 that is
    equivalent to the mass M2 is used because it better captures the behaviour of the system based on the relative
    sizes of the two large bodies.

    :param m1: Mass of the largest object in the simulated 3 body system
    :param m2: Decimal ratio of the mass M2 to M1
    :param m2_orbital_radius: orbital radius of the object of mass M2
    :param x_points: x-coordinates of the grid
    :param y_points: y-coordinates of the grid
    :return: None
    """

    gravitational_constant = 1
    m2_absolute = m2 * m1

    omega = sqrt(3 * gravitational_constant * m2_absolute / m2_orbital_radius)

    # Note that since M1/(M1+M2) gets reduced to 2/3 and M2/(M1+M2) gets reduced to 1/3
    # Due to our earlier definitions for M1 and M2                                                            #
    # Also note that Z as defined above will result in an indentation error,
    # but was typed this way, so it could fit on one page
    gravitational_potential = -0.5 * (omega ** 2) * (x_points ** 2 + y_points ** 2)
    -(2 * gravitational_constant * m2_absolute) / (sqrt((m2_orbital_radius / 3 - x_points) ** 2 + y_points ** 2))
    -(gravitational_constant * m2_absolute) / (sqrt((m2_orbital_radius * 2 / 3 - x_points) ** 2 + y_points ** 2))

    return gravitational_potential


if __name__ == "__main__":
    # Function calls for plot_gravity_potential
    M1 = 2
    M2 = 0.5
    R = 1

    # Defining the x,y coordinates of our plot
    plot_length = 200
    grid_x, grid_y = meshgrid(linspace(-1.5, 1.5, plot_length), linspace(-1.5, 1.5, plot_length))

    # Configure logger
    log = getLogger()
    log.setLevel(DEBUG)
    log.addHandler(StreamHandler(stdout))

    start_elapsed = perf_counter()
    start_process = process_time()
    grid_z = gravity_potential(M1, M2, R, grid_x, grid_y)
    figure(figsize=(10, 10), dpi=100)
    contour(grid_x, grid_y, grid_z, 500)
    colorbar()
    show()
    end_elapsed = perf_counter()
    end_process = process_time()
    info(f"Elapsed time: {end_elapsed - start_elapsed}")
    info(f"Process time: {end_process - start_process}")

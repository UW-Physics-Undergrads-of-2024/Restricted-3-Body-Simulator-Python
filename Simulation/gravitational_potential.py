from logging import DEBUG, getLogger, StreamHandler
from matplotlib.pyplot import figure, contour, colorbar, show
from numba import jit
from numpy import meshgrid, linspace, sqrt, array
from sys import stdout
from time import process_time, perf_counter


@jit(nopython=True)
def gravity_potential(m1: float, m2: float, m2_orbital_radius: float,
                      x_points: array, y_points: array) -> array:
    """
    Takes an array of coordinates from a numpy.meshgrid object and determines the ratio of the gravitational potential
    and the mass of the small object at each point in the meshgrid. The ratio is used rather than the potential field
    itself because the resulting field will be something that holds true for any small third object in the three body
    system, rather generating a similarly shaped field but with varying values proportional to the mass of the third
    object.

    :param m1: Mass of the largest object in the simulated 3 body system
    :param m2: Mass of the second-largest object in the simulated 3 body system
    :param m2_orbital_radius: orbital radius of the object of mass M2
    :param x_points: x-coordinates of the grid
    :param y_points: y-coordinates of the grid

    :return: None

    :raise ValueError: Either of the masses is non-positive or m2 is larger than m1
    """

    if m1 <= 0:
        raise ValueError(f"The value of m1 must be positive, received {m1}")

    if m2 <= 0:
        raise ValueError(f"The value of m2 must be positive, received {m2}")

    if m2 > m1:
        raise ValueError(f" The value of m1 must be positive but less than or equal to m2.\n"
                         f" I.e. the ratio of m2 to m1 must be in (0, 1], received {m2/m1}")

    gravitational_constant = 1

    # By convention,  omega is used as the variable representing angular frequency
    omega = sqrt(gravitational_constant * (m1 + m2) / m2_orbital_radius**3)

    gravitational_potential = -0.5 * (omega ** 2) * (x_points ** 2 + y_points ** 2) - gravitational_constant * (
            m2 / (sqrt((m2 * m2_orbital_radius / (m1 + m2) - x_points)**2 + y_points**2))
            + m2 / (sqrt((m1 * m2_orbital_radius / (m1 + m2) + x_points)**2 + y_points**2)))

    return gravitational_potential


def main() -> None:
    # Function calls for plot_gravity_potential
    large_mass = 2
    second_mass = 0.5
    orbital_radius = 1

    # Defining the x,y coordinates of our plot
    plot_length = 200
    grid_x, grid_y = meshgrid(linspace(-1.5, 1.5, plot_length), linspace(-1.5, 1.5, plot_length))

    # Configure logger, with the scope name used as the logger name
    log = getLogger(__name__)
    log.setLevel(DEBUG)
    log.addHandler(StreamHandler(stdout))

    start_elapsed = perf_counter()
    start_process = process_time()
    grid_z = gravity_potential(large_mass, second_mass, orbital_radius, grid_x, grid_y)
    figure(figsize=(10, 10), dpi=100)
    contour(grid_x, grid_y, grid_z, 500)
    colorbar()
    show()
    end_elapsed = perf_counter()
    end_process = process_time()
    log.info(f"Elapsed time: {end_elapsed - start_elapsed}")
    log.info(f"Process time: {end_process - start_process}")


if __name__ == "__main__":
    main()

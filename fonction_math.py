import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp

def euclidean_distance(city_coords, satellite_coords):
    """
    Calculate the Euclidean distance between a city and a satellite in a 3D space.

    Parameters:
    - city_coords (numpy array): A tuple containing the x, y, and z coordinates of the city.
    - satellite_coords (numpy array): A tuple containing the x, y, and z coordinates of the satellite.

    Returns:
    float: The Euclidean distance between the city and the satellite.
    """
    delta = city_coords - satellite_coords
    return math.sqrt(np.dot(delta, delta))


def total_intensity(city_coords, satellites_coords, satellite_power):
    """
    Calculate the total intensity at a city location based on an inverse square law.

    Parameters:
    - city_coords (numpy array): A tuple containing the x, y, and z coordinates of the city.
    - satellites_coords (numpy array): A list of numpy arrays, each containing the x, y, and z coordinates of a satellite.
    - satellite_power (float): Power of each satellite signal.

    Returns:
    float: The total intensity at the city location.
    """
    sum_inverse_dist_squared = 0
    for satellite_coord in satellites_coords:
        sum_inverse_dist_squared += 1 / euclidean_distance(city_coords, satellite_coord) ** 2
    return sum_inverse_dist_squared * satellite_power


def city_intensities(cities_coords, satellites_coords, satellite_power):
    intensities = np.zeros(np.shape(cities_coords)[0])
    for i, city_coord in enumerate(cities_coords):
        intensities[i] = total_intensity(city_coord, satellites_coords, satellite_power)
    return intensities


def is_covered_3D(city_coords, satellites_coords):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        surface_radius = math.sqrt(satellite_coords[3] ** 2 - satellite_coords[2] ** 2)  # Pythagora sqrt(r^2 - h^2)
        city_satellite_distance = math.sqrt(
            (city_coords[0] - satellite_coords[0]) ** 2 + (city_coords[1] - satellite_coords[1]) ** 2)
        # print(surface_radius)
        # print(city_satellite_distance)
        if surface_radius >= city_satellite_distance:
            return True
    return False


def plot_covering_3D(cities_coordinates, satellites_coordinates):
    # Extraire les coordonnées x et y des points
    x_coords, y_coords, z_coords = zip(*cities_coordinates)

    # Tracer les points
    for x_coord, y_coord in zip(x_coords, y_coords):
        plt.scatter(x_coord, y_coord,
                    color='green' if is_covered_3D(np.array([x_coord, y_coord]), satellites_coordinates) else 'red',
                    marker='o')

    # Tracer les cercles
    for center, radius in zip(satellites_coordinates[:, :2],
                              np.sqrt(satellites_coordinates[:, 3] ** 2 - satellites_coordinates[:, 2] ** 2)):
        circle = Circle(center, radius, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(circle)

    plt.title('Network coverage')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.axis('equal')

    plt.show()


def is_covered_2D(city_coords, satellites_coords):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        city_satellite_distance = math.sqrt(
            (city_coords[0] - satellite_coords[0]) ** 2 + (city_coords[1] - satellite_coords[1]) ** 2)
        if satellite_coords[2] >= city_satellite_distance:
            return True
    return False


def plot_covering_2D(cities_coordinates, satellites_coordinates, grid_size):
    # Extraire les coordonnées x et y des points
    x_coords, y_coords = zip(*cities_coordinates)

    # cercles
    for center, radius in zip(satellites_coordinates[:, :2], satellites_coordinates[:, 2]):
        circle = Circle(center, radius, edgecolor='blue', facecolor='none', label='Satellites')
        plt.gca().add_patch(circle)
    # villes
    for x_coord, y_coord in zip(x_coords, y_coords):
        plt.scatter(x_coord, y_coord,
                    color='green' if is_covered_2D(np.array([x_coord, y_coord]), satellites_coordinates) else 'red',
                    marker='o')

    num_satellites = len(satellites_coordinates)

    # satellites
    plt.scatter(satellites_coordinates[:, 0], satellites_coordinates[:, 1], color='blue', marker='x',
                label='Satellites')

    plt.title(f'Network coverage of {len(cities_coordinates)} cities by {len(satellites_coordinates)} satellites for a radius of {radius}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')

    plt.grid(color='gray', linestyle='dashed')

    plt.show()


"""city_A = np.array([1, 2, 0]) # x, y, z
city_B = np.array([4, 8, 0])
city_C = np.array([5, 2, 0])
city_D = np.array([8, 14, 0])
city_E = np.array([10, 6, 0])
satellite_A = np.array([2, 5, 12, 13]) # x, y, z, range
satellite_B = np.array([4, 3, 12, 13])
satellite_C = np.array([11, 8, 12, 12.5])
cities_coordinates = np.array([city_A, city_B, city_C, city_D, city_E])
satellites_coordinates = np.array([satellite_A, satellite_B, satellite_C])"""





def solve_2D(cities_coordinates, grid_size, radius, num_satellites):
    num_cities = cities_coordinates.shape[0]

    # Create a matrix for distances
    distances_matrix = np.zeros((num_cities, (grid_size + 1) ** 2))

    for i in range(num_cities):
        city_x, city_y = cities_coordinates[i]
        for j in range((grid_size + 1) ** 2):  # (0,0) puis (0,1) puis (0,2) ...
            x_pos = j // (grid_size + 1)
            y_pos = j % (grid_size + 1)
            distances_matrix[i, j] = np.linalg.norm([city_x - x_pos, city_y - y_pos])

    # Variables
    satellite_positions = cp.Variable((grid_size + 1) ** 2, boolean=True)

    # Objective
    objective = cp.Minimize(cp.sum(satellite_positions))

    # Constraints
    constraints = []
    for i in range(num_cities):
        indices = []
        for j in range((grid_size + 1) ** 2):
            if distances_matrix[i, j] <= radius:
                indices.append(j)

        city_coverage = cp.sum(satellite_positions[indices])
        constraints.append(city_coverage >= 1)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    solution_matrix = satellite_positions.value.astype(int).reshape(grid_size + 1, grid_size + 1)

    satellite_indices = np.argwhere(solution_matrix == 1)
    num_satellites = satellite_indices.shape[0]  # Nombre de satellites placés
    satellites_coordinates = np.hstack((satellite_indices, np.full((num_satellites, 1), radius)))
    print(satellites_coordinates)
    # Results
    print("Optimal satellite positions:")
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2
    print(solution_matrix)
    print("Optimal number of satellites: {}".format(problem.value))

    return satellites_coordinates

def is_in_radius(city, satellite,radius, grid_size):
    if (city[0] - satellite[0]) ** 2 + (city[1] - satellite[1]) ** 2 <= radius ** 2:
        return True
    return False

def solve_2D_ws(cities_coordinates, grid_size, radius, num_satellites):

    # Variables
    satellite_positions = cp.Variable((grid_size + 1) ** 2, boolean=True)
    is_covered = cp.Variable(len(cities_coordinates),  boolean=True)

    # Objective
    objective = cp.Maximize(cp.sum(is_covered))

    # contraintes
    constraints = []
    constraints.append([cp.sum(satellite_positions) == num_satellites])

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    

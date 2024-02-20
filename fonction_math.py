import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random

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

def create_weight(n_cities):
    # take the number of cities
    # return a random list of weights with 2 decimals so that the sum of the weights is 1
    poids = list()
    rest = 1
    i = 0
    while i < n_cities - 1 :
        p = random.uniform(0.01, rest/2)
        poids.append(p)
        rest = rest - poids[i]
        i = i + 1
    poids.append(rest)


    for i in range(n_cities):
        poids[i] = round(poids[i],2)

    
    poids = np.array(poids)
    diff = 1 - np.sum(poids)
    poids[0] = poids[0] + diff
    return poids




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


def plot_covering_2D(cities_coordinates, cities_weights, satellites_coordinates, grid_size):
    # Extraire les coordonnées x et y des points
    x_coords, y_coords = zip(*cities_coordinates)

    # Villes
    population_proportion = 0
    for x_coord, y_coord, weight in zip(x_coords, y_coords, cities_weights):
        is_covered = is_covered_2D(np.array([x_coord, y_coord]), satellites_coordinates)
        if is_covered:
            population_proportion += weight
        plt.scatter(x_coord, y_coord,
                    color='green' if is_covered else 'red',
                    marker='o',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8)
    
    # Satellites
    for center, radius in zip(satellites_coordinates[:, :2], satellites_coordinates[:, 2]):
        circle = Circle(center, radius, color='red', alpha=0.1)
        plt.gca().add_patch(circle)

    plt.text(0.85, -0.1, f'Population Proportion: {np.round(population_proportion, decimals=4)}',
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)
    # mettre le première arg à 0.1 pour afficher en bas à gauche
        


    plt.scatter(satellites_coordinates[:, 0], satellites_coordinates[:, 1], color='blue', marker='x')

    radius = round(radius, 2)
    plt.title(f'Network coverage of {len(cities_coordinates)} cities by {len(satellites_coordinates)} satellites for a radius of {radius}', fontweight='bold')  # Increase the title font size and weight
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
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





def is_in_radius(city, satellite,radius, grid_size):
    if (city[0] - satellite[0]) ** 2 + (city[1] - satellite[1]) ** 2 <= radius ** 2:
        return True
    return False



def distance_angulaire(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1);lon1 = np.radians(lon1);lat2 = np.radians(lat2);lon2 = np.radians(lon2)
    a=np.array([np.cos(lat1)*np.cos(lon1),np.cos(lat1)*np.sin(lon1),np.sin(lat1)])
    b=np.array([np.cos(lat2)*np.cos(lon2),np.cos(lat2)*np.sin(lon2),np.sin(lat2)])
    return np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def solve_2D(N_satellites, cities_coordinates, cities_weights, grid_size = 10, scope = 15, height = 4, intensity = 1000):
    num_cities = cities_coordinates.shape[0]
    radius = np.sqrt(scope**2 - height**2)

    # Create a matrix for distances
    grid_points = np.array(np.meshgrid(np.arange(grid_size + 1), np.arange(grid_size + 1))).T.reshape(-1, 2)
    distances_matrix = np.linalg.norm(cities_coordinates[:, np.newaxis, :] - grid_points, axis=2) # sqrt(x^2 + y^2)
    distances_matrix = np.sqrt(np.square(distances_matrix) + np.square(height)) # sqrt(x^2 + y^2 + height^2)
    inv_squared_distances_matrix = intensity / (4*math.pi*np.square(distances_matrix))

    # Variables
    satellite_positions = cp.Variable((grid_size + 1)**2, boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)
    how_many_times_covered = cp.Variable(num_cities, integer=True)

    # Objective
    objective = cp.Maximize(cp.sum(inv_squared_distances_matrix @ satellite_positions))
    #objective = cp.Maximize(cp.sum(how_many_times_covered))

    indices_within_scope = [
        np.where(distances_matrix[i] <= scope)[0] for i in range(num_cities)
    ]

    # Constraints
    constraints = []

    constraints.append(cp.sum(satellite_positions) == N_satellites)
    constraints.append(cp.sum(city_covered @ cities_weights) >= 0.8)
    for i in range(num_cities):
        constraints.append(how_many_times_covered[i] == cp.sum(satellite_positions[indices_within_scope[i]]))
        constraints.append(city_covered[i] >= how_many_times_covered[i]/len(indices_within_scope[i]))
        constraints.append(city_covered[i] <= how_many_times_covered[i])

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI, warm_start=True)

    if problem.status != cp.OPTIMAL:
        raise Exception("The problem is not solvable")

    # Results
    print("Part de la population ayant accès au réseau")
    print(cp.sum(city_covered @ cities_weights).value)
    print("Positions des satellites")
    print(satellite_positions.value)
    print("Villes couvertes")
    print(city_covered.value)
    print("Valeur de l'objectif")
    print(problem.value)
    print("Nombre de fois où chaque ville est couverte:")
    print(how_many_times_covered.value)
    solution_matrix = satellite_positions.value.astype(int).reshape(grid_size+1,grid_size+1)
    coords = np.argwhere(solution_matrix == 1)
    coords_avec_rayon = np.c_[coords, np.full((len(coords), 1), radius)]
    print("Positions optimales des satellites:")
    print(coords_avec_rayon)

    return coords_avec_rayon
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random
from sklearn.cluster import KMeans

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

def nbr_max_sat(cities_coordinates, grid_size, radius):
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
    # Results
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2

    return len(satellites_coordinates)

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

def is_in_radius(city, satellite,radius, grid_size):
    if (city[0] - satellite[0]) ** 2 + (city[1] - satellite[1]) ** 2 <= radius ** 2:
        return True
    return False


def distance_angulaire(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1);lon1 = np.radians(lon1);lat2 = np.radians(lat2);lon2 = np.radians(lon2)
    a=np.array([np.cos(lat1)*np.cos(lon1),np.cos(lat1)*np.sin(lon1),np.sin(lat1)])
    b=np.array([np.cos(lat2)*np.cos(lon2),np.cos(lat2)*np.sin(lon2),np.sin(lat2)])
    return np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Parameters:
    - coord1 (numpy array): A tuple containing the latitude and longitude of the first point.
    - coord2 (numpy array): A tuple containing the latitude and longitude of the second point.

    Returns:
    float: The distance between the two points in kilometers.
    """
    # convert decimal degrees to radians
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def k_means_cities(cities_coordinates, k, cities_weights):
    """
    Apply the k-means algorithm to the cities' coordinates
    Args:
        cities_coordinates: les coordonnées des villes
        k: le nombre de centroïdes

    Returns:

    """
    kmeans = KMeans(n_clusters=k )
    kmeans.fit(cities_coordinates, cities_weights)
    #retourne les nouveaux poids
    new_weights = np.array([0 for i in range(k)], dtype=float)
    prediction = kmeans.predict(cities_coordinates)
    for i in range(len(cities_coordinates)):
        new_weights[int(prediction[i])] += cities_weights[i]
    return kmeans.cluster_centers_, new_weights

#petit test mignon pas du tout à sa place qui plot les villes et les nouveaux centroïdes, et colorie de la même couleur les clustered cities
def plot_kmeans(cities_coordinates, k, new_centroids):
    plt.scatter(cities_coordinates[:, 0], cities_coordinates[:, 1], c='black')
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red')
    plt.show()
"""
cities_coordinates = np.array([[1, 2],[0,1],[1,0],[2,1]])
cities_weights = np.array([0.4,0.4,0.1,0.1])
k = 3
new_centroids, new_weights = k_means_cities(cities_coordinates, k, cities_weights)
plot_kmeans(cities_coordinates, k, new_centroids)
print(new_weights)

"""
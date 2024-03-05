import fonction_math as fm
import numpy as np
import cvxpy as cp
import math

def spherical_satellites_repartition( N_satellites ,cities_coordinates,  cities_weights):

    """
    ARGS: 
    - N_satellites (int): Number of satellites
    - cities_coordinates (numpy array): A list of numpy arrays, each containing the latitude and longitude of a city.
    - cities_weights (numpy array): A list of weights for each city. (sum of weights = 1)

    RETURNS:
    - numpy array: A list of numpy arrays, each containing the latitude, longitude and altitude of a satellite.

    """
    num_cities = cities_coordinates.shape[0]
    radius_earth = 6371  # km
    radius_satellite = 35786  # km
    height_satellite = 35786 - 6371  # km
    # Create a matrix for distances
    distances_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distances_matrix[i, j] = fm.haversine(cities_coordinates[i], cities_coordinates[j])
    inv_squared_distances_matrix = 1 / (4*math.pi*np.square(distances_matrix))
    # Variables
    satellite_positions = cp.Variable(num_cities, boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)
    # Objective
    objective = cp.Maximize(cp.sum(inv_squared_distances_matrix @ satellite_positions))
    # Constraints
    constraints = []
    constraints.append(cp.sum(satellite_positions) == N_satellites)
    constraints.append(cp.sum(city_covered @ cities_weights) >= 0.8)
    for i in range(num_cities):
        constraints.append(city_covered[i] <= cp.sum(satellite_positions * inv_squared_distances_matrix[i]))
        constraints.append(city_covered[i] >= cp.sum(satellite_positions * inv_squared_distances_matrix[i]) / num_cities)
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI, warm_start=True)
    if problem.status != cp.OPTIMAL:
        return np.array([])
    # Results
    satellites_coordinates = []
    for i in range(num_cities):
        if satellite_positions.value[i] > 0.5:
            lat = cities_coordinates[i][0]
            long = cities_coordinates[i][1]
            alt = radius_earth + height_satellite
            satellites_coordinates.append([lat, long, alt])
    return np.array(satellites_coordinates)
    
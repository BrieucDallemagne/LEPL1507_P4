import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random
import fonction_math as fm

def solve_2D_v1(N_satellites, cities_coordinates, cities_weights, grid_size = 10, radius = 3):
    num_cities = cities_coordinates.shape[0]

    # Create a matrix for distances
    distances_matrix = np.zeros((num_cities, (grid_size + 1)**2))

    for i in range(num_cities):
        city_x, city_y = cities_coordinates[i]
        for j in range((grid_size + 1)**2): # (0,0) puis (0,1) puis (0,2) ...
            x_pos = j // (grid_size + 1)
            y_pos = j % (grid_size + 1)
            distances_matrix[i, j] = np.linalg.norm([city_x - x_pos, city_y - y_pos])

    # Variables
    satellite_positions = cp.Variable((grid_size + 1)**2, boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)

    # Objective
    objective = cp.Maximize(cp.sum(city_covered))

    # Constraints
    constraints = []
    for i in range(num_cities):
        indices = []
        for j in range((grid_size + 1)**2):
            if distances_matrix[i, j] <= radius:
                indices.append(j)
        constraints.append(city_covered[i] == cp.sum(satellite_positions[indices]))
    
    constraints.append(cp.sum(satellite_positions) == N_satellites)
    
    

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    solution_matrix = satellite_positions.value.astype(int).reshape(grid_size+1,grid_size+1)


    # Results
    coords = np.argwhere(solution_matrix == 1)
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1 or solution_matrix[city_coordinates[0], city_coordinates[1]] == 3):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2

    
    coords_avec_colonne = np.c_[coords, np.full((len(coords), 1), radius)]

    return coords_avec_colonne


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


def solve_2D_v2(N_satellites, cities_coordinates, cities_weights, grid_size = 10, radius = 3):
    num_cities = cities_coordinates.shape[0]

    # Create a matrix for distances
    distances_matrix = np.zeros((num_cities, (grid_size + 1)**2))

    for i in range(num_cities):
        city_x, city_y = cities_coordinates[i]
        for j in range((grid_size + 1)**2): # (0,0) puis (0,1) puis (0,2) ...
            x_pos = j // (grid_size + 1)
            y_pos = j % (grid_size + 1)
            distances_matrix[i, j] = np.linalg.norm([city_x - x_pos, city_y - y_pos])

    # Variables
    satellite_positions = cp.Variable((grid_size + 1)**2, boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)

    # Objective
    objective = cp.Maximize(cp.sum(city_covered))

    # Constraints
    constraints = []
    for i in range(num_cities):
        indices = []
        for j in range((grid_size + 1)**2):
            if distances_matrix[i, j] <= radius:
                indices.append(j)
        constraints.append(city_covered[i] == cp.sum(satellite_positions[indices]))
    
    constraints.append(cp.sum(satellite_positions) == N_satellites)
    
    

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    solution_matrix = satellite_positions.value.astype(int).reshape(grid_size+1,grid_size+1)


    # Results
    coords = np.argwhere(solution_matrix == 1)
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1 or solution_matrix[city_coordinates[0], city_coordinates[1]] == 3):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2
    
    coords_avec_colonne = np.c_[coords, np.full((len(coords), 1), radius)]

    return coords_avec_colonne


def old_solver( N_satellites ,cities_coordinates,  cities_weights):

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






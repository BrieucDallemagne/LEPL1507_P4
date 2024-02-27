import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random

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
    print("Optimal satellite positions:")
    coords = np.argwhere(solution_matrix == 1)
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1 or solution_matrix[city_coordinates[0], city_coordinates[1]] == 3):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2

    

    print(solution_matrix)
    
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
    print("Optimal satellite positions:")
    coords = np.argwhere(solution_matrix == 1)
    for city_coordinates in cities_coordinates:
        if (solution_matrix[city_coordinates[0], city_coordinates[1]] == 1 or solution_matrix[city_coordinates[0], city_coordinates[1]] == 3):
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 3
        else:
            solution_matrix[city_coordinates[0], city_coordinates[1]] = 2

    

    print(solution_matrix)
    
    coords_avec_colonne = np.c_[coords, np.full((len(coords), 1), radius)]

    return coords_avec_colonne



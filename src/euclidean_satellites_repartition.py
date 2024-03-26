import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fonction_math as fm
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random

def euclidean_satellites_répartition(N_satellites, cities_coordinates, cities_weights, grid_size = 10, scope = 15, height = 4, intensity = 1000):
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
    
    solution_matrix = satellite_positions.value.astype(int).reshape(grid_size+1,grid_size+1)
    coords = np.argwhere(solution_matrix == 1)
    coords_avec_rayon = np.c_[coords, np.full((len(coords), 1), radius)]

    return coords_avec_rayon


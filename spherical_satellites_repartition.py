import fonction_math as fm
import numpy as np
import cvxpy as cp
import math
from scipy.spatial.distance import cdist

def spherical_satellites_repartition(cities_coordinates, cities_weights, scope = 15, height = 4):
    num_cities = cities_coordinates.shape[0]

    earth_radius = 50
    satellite_radius = earth_radius + height
    sphere_center = (0, 0, 0)

    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    # Create 2D arrays for theta and phi
    phi, theta = np.meshgrid(phi_values, theta_values)
    print(theta)
    print(phi)

    # Calculate x, y, and z coordinates using vectorized operations
    x_grid = (sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)).flatten()
    y_grid = (sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)).flatten()
    z_grid = (sphere_center[2] + satellite_radius * np.cos(phi)).flatten()

    # Create a matrix for distances
    #grid_points = np.array(np.meshgrid(np.arange(grid_size + 1), np.arange(grid_size + 1))).T.reshape(-1, 2)
    grid_points = np.column_stack((x_grid, y_grid, z_grid))
    print(grid_points)
    distances_matrix = cdist(cities_coordinates, grid_points)
    print(distances_matrix)
    #distances_matrix = np.sqrt(np.square(distances_matrix) + np.square(height)) # sqrt(x^2 + y^2 + height^2)
    inv_squared_distances_matrix = 1 / (4*math.pi*np.square(distances_matrix))

    # Variables
    satellite_positions = cp.Variable(len(theta_values)*len(phi_values), boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)
    how_many_times_covered = cp.Variable(num_cities, integer=True)
    
    # Objective
    objective = cp.Minimize(cp.sum(satellite_positions))
    #objective = cp.Maximize(cp.sum(how_many_times_covered))

    indices_within_scope = [
        np.where(distances_matrix[i] <= scope)[0] for i in range(num_cities)
    ]
    print("Indices within scope")
    print(indices_within_scope)

    # Constraints
    constraints = []

    constraints.append((city_covered @ cities_weights) >= 0.8)
    for i in range(num_cities):
        constraints.append(how_many_times_covered[i] == cp.sum(satellite_positions[indices_within_scope[i]]))
        constraints.append(city_covered[i] >= how_many_times_covered[i]/(len(theta)*len(phi)))
        constraints.append(city_covered[i] <= how_many_times_covered[i])

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI, warm_start=True)


    # Results
    print("Part de la population ayant accès au réseau")
    print(cp.sum(city_covered @ cities_weights).value)
    print("Positions des satellites")
    print(satellite_positions.value)
    print("Villes couvertes")
    print(city_covered.value)
    print("Valeur de l'objectif")
    print(problem.value)
    print("Intensités des villes")
    print((inv_squared_distances_matrix @ satellite_positions).value)
    print("Nombre de fois où chaque ville est couverte")
    print(how_many_times_covered.value)
    solution_matrix = satellite_positions.value.astype(int).reshape(len(theta_values), len(phi_values))
    print("Solution matrix")
    print(solution_matrix)
    where = np.argwhere(solution_matrix == 1)
    print(where)
    coords = np.array((theta_values[where[:, 0]], phi_values[where[:, 1]])).T
    print("Coords")
    print(coords)
    """coords_avec_rayon = np.c_[coords, np.full((len(coords), 1), radius)]
    print("Positions optimales des satellites")
    print(coords_avec_rayon)"""

    return coords
import fonction_math as fm
import numpy as np
import cvxpy as cp
import math
from scipy.spatial.distance import cdist


def spherical_satellites_repartition_old(cities_coordinates, cities_weights, height=4, verbose=False):
    num_cities = cities_coordinates.shape[0]

    earth_radius = 50
    scope = fm.find_x(height, earth_radius)
    satellite_radius = earth_radius + height
    sphere_center = (0, 0, 0)

    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    # Create 2D arrays for theta and phi
    phi, theta = np.meshgrid(phi_values, theta_values)

    # Calculate x, y, and z coordinates using vectorized operations
    x_grid = (sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)).flatten()
    y_grid = (sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)).flatten()
    z_grid = (sphere_center[2] + satellite_radius * np.cos(phi)).flatten()

    # Create a matrix for distances
    grid_points = np.column_stack((x_grid, y_grid, z_grid))
    distances_matrix = cdist(cities_coordinates, grid_points)
    inv_squared_distances_matrix = 1 / (4 * math.pi * np.square(distances_matrix))

    # Variables
    satellite_positions = cp.Variable(len(theta_values) * len(phi_values), boolean=True)
    city_covered = cp.Variable(num_cities, boolean=True)
    how_many_times_covered = cp.Variable(num_cities, integer=True)

    # Objective
    objective = cp.Minimize(cp.sum(satellite_positions))
    # objective = cp.Maximize(cp.sum(how_many_times_covered))

    indices_within_scope = [
        np.where(distances_matrix[i] <= scope)[0] for i in range(num_cities)
    ]

    # Constraints
    constraints = []

    constraints.append((city_covered @ cities_weights) >= 0.8)
    for i in range(num_cities):
        constraints.append(how_many_times_covered[i] == cp.sum(satellite_positions[indices_within_scope[i]]))
        constraints.append(city_covered[i] >= how_many_times_covered[i] / (len(theta) * len(phi)))
        constraints.append(city_covered[i] <= how_many_times_covered[i])
    #constraints.append(inv_squared_distances_matrix @ satellite_positions >= np.full(num_cities, 1e-3))

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI, warm_start=True)
    solution_matrix = satellite_positions.value.astype(int).reshape(len(theta_values), len(phi_values))

    where = np.argwhere(solution_matrix == 1)

    coords = np.array((theta_values[where[:, 0]], phi_values[where[:, 1]])).T

    if verbose:
        print("Part de la population ayant accès au réseau")
        print(cp.sum(city_covered @ cities_weights).value)
        print("Positions des satellites")
        print(satellite_positions.value)
        print("Villes couvertes")
        print(city_covered.value)
        print("Valeur de l'objectif (nombre de satellites minimum)")
        print(problem.value)
        print("Intensités des villes")
        print((inv_squared_distances_matrix @ satellite_positions).value)
        print("Nombre de fois où chaque ville est couverte")
        print(how_many_times_covered.value)
        print("Solution matrix")
        print(solution_matrix)
        print("Coordonnées des satellites (theta, phi)")
        print(coords)

    return coords

def spherical_satellites_repartition(cities_coordinates, cities_weights, height=4, verbose=False):
    num_cities = cities_coordinates.shape[0]

    earth_radius = 50
    scope = fm.find_x(height, earth_radius)
    satellite_radius = earth_radius + height
    sphere_center = (0, 0, 0)

    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    # Create 2D arrays for theta and phi
    phi, theta = np.meshgrid(phi_values, theta_values)

    # Calculate x, y, and z coordinates using vectorized operations
    x_grid = (sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)).flatten()
    y_grid = (sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)).flatten()
    z_grid = (sphere_center[2] + satellite_radius * np.cos(phi)).flatten()

    # Create a matrix for distances
    grid_points = np.column_stack((x_grid, y_grid, z_grid))
    distances_matrix = cdist(cities_coordinates, grid_points)
    inv_squared_distances_matrix = fm.I(distances_matrix)#1 / (np.square(distances_matrix))    

    # Variables
    satellite_positions = cp.Variable(len(theta_values) * len(phi_values), boolean=True)
    enough_intensity = cp.Variable(num_cities, boolean=True)

    # Objective
    objective = cp.Minimize(cp.sum(satellite_positions))
    # objective = cp.Maximize(cp.sum(how_many_times_covered))

    indices_within_scope = [
        np.where(distances_matrix[i] <= scope)[0] for i in range(num_cities)
    ]

    # Constraints
    constraints = []
    min_intensity = fm.minimum_intensity(height,earth_radius,fm.I)[0] #fm.minimum_intensity(height) 

    constraints.append((enough_intensity @ cities_weights) >= 0.8)
    for i in range(num_cities):
        intensity = cp.sum(cp.multiply(inv_squared_distances_matrix[i], satellite_positions)[indices_within_scope[i]])
        constraints.append(enough_intensity[i] <= intensity/min_intensity)
        constraints.append(min_intensity - intensity >= -1000000*enough_intensity[i])
        

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI, warm_start=True, verbose=False)
    solution_matrix = satellite_positions.value.astype(int).reshape(len(theta_values), len(phi_values))

    where = np.argwhere(solution_matrix == 1)

    coords = np.array((theta_values[where[:, 0]], phi_values[where[:, 1]])).T

    if verbose:
        print("Part de la population ayant accès au réseau")
        print(cp.sum(enough_intensity @ cities_weights).value)
        print("Positions des satellites")
        print(satellite_positions.value)
        print("Couverture acceptable")
        print(enough_intensity.value)
        print("Valeur de l'objectif (nombre de satellites minimum)")
        print(problem.value)
        cities_intensity = []
        for i in range(num_cities):
            cities_intensity.append(cp.sum(cp.multiply(inv_squared_distances_matrix[i], satellite_positions)[indices_within_scope[i]]).value)
        print("Intensités des villes")
        print(cities_intensity)
        #print((inv_squared_distances_matrix @ satellite_positions).value)
        print("Solution matrix")
        print(solution_matrix)
        print("Coordonnées des satellites (theta, phi)")
        print(coords)

    return coords
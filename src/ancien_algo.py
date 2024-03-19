import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import plots.plot_rond as pr
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


def plot_3D_old(cities_coordinates, satellites_coordinates, cities_weights, height, original_cities_coordinates=np.array(0), original_cities_weights=np.array(0), kmeans=False,rot=False):
    sphere_center = (0, 0, 0)
    earth_radius = 50

    # Créer le plot en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rayon de la sphère
    earth_radius = 50
    satellite_radius = 50 + height
    scope = fm.find_x(height, earth_radius)

    # Créer la terre
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = sphere_center[0] + earth_radius * np.outer(np.cos(u), np.sin(v))
    y = sphere_center[1] + earth_radius * np.outer(np.sin(u), np.sin(v))
    z = sphere_center[2] + earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Dessiner la sphère
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    """# Dessiner le quadrillage
    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    theta, phi = np.meshgrid(theta_values, phi_values)
    x_grid = sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)
    y_grid = sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)
    z_grid = sphere_center[2] + satellite_radius * np.cos(phi)

    ax.plot_wireframe(x_grid, y_grid, z_grid, color='black', linewidth=0.5)

    ax.scatter(x_grid, y_grid, z_grid, color='red', s=10, alpha=0.2)"""

    x_sat = sphere_center[0] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.cos(
        satellites_coordinates[:, 0])
    y_sat = sphere_center[1] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.sin(
        satellites_coordinates[:, 0])
    z_sat = sphere_center[2] + satellite_radius * np.cos(satellites_coordinates[:, 1])

    satellites_spherical_coordinates = np.c_[x_sat, y_sat, z_sat]

    ax.scatter(x_sat, y_sat, z_sat, color='blue', s=40, label='Satellites')
    for x_s, y_s, z_s in zip(x_sat, y_sat, z_sat):
        x = x_s + scope * np.outer(np.cos(u), np.sin(v))
        y = y_s + scope * np.outer(np.sin(u), np.sin(v))
        z = z_s + scope * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    for x_city, y_city, z_city in cities_coordinates:
        is_covered = pr.is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
        ax.scatter(x_city, y_city, z_city, c='green' if pr.is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and pr.has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.inten_min(height, earth_radius, fm.I)[0], scope) else
                                            "orange" if pr.is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and not pr.has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.inten_min(height, earth_radius, fm.I)[0], scope) else
                                            "red", s=20, marker='o')
    i = 0
    if kmeans:
        for x_city, y_city, z_city in original_cities_coordinates:
            is_covered = pr.is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
            ax.scatter(x_city, y_city, z_city, c='pink' if is_covered else "orange", s=20, marker='o')
            ax.text(x_city, y_city, z_city, '%s' % (str(original_cities_weights[i])), size=5, zorder=1,
                    color='k')
            i += 1

    # Configurer les limites de l'axe
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_zlim(-70, 70)

    # Make all axes equal in size
    ax.set_box_aspect([1, 1, 1])

    # Ajouter une légende
    ax.legend()

    # Afficher le plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if rot:
        def animate(frame):
            ax.view_init(30, 6 * frame)
            plt.pause(.001)
            return fig

        anim = pr.animation.FuncAnimation(fig, animate, frames=200, interval=100)
    plt.show()






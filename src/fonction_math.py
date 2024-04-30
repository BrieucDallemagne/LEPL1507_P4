import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
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

def k_means_cities(cities_coordinates, k, cities_weights):
    """
    Apply the k-means algorithm to the cities' coordinates
    Args:
        cities_coordinates: les coordonnées des villes
        k: le nombre de centroïdes

    Returns:

    """
    cities_coordinates = spherical_to_cartesian(cities_coordinates, [0, 0, 0], 50)
    kmeans = KMeans(n_clusters=k, max_iter = 100)
    kmeans.fit(cities_coordinates, cities_weights)
    #retourne les nouveaux poids
    new_weights = np.array([0 for i in range(k)], dtype=float)
    prediction = kmeans.predict(cities_coordinates)
    for i in range(len(cities_coordinates)):
        new_weights[int(prediction[i])] += cities_weights[i]
    new_coordinates = kmeans.cluster_centers_
    for i in range(k):
        new_coordinates[i] = new_coordinates[i] / np.linalg.norm(new_coordinates[i]) * 50
    return new_coordinates, new_weights

def adapt_to_3D(cities_en_deux_d, val_array):
    ret = np.zeros((len(cities_en_deux_d), 3))
    for i in range(len(cities_en_deux_d)):
        #ajoute un 0 à chaque sous array: [1,2] devient [1,2,0]
        ret[i] = np.append(cities_en_deux_d[i], val_array[i])
    return ret

def supp_3D(cities_en_trois_d):
    ret = np.zeros((len(cities_en_trois_d), 2))
    for i in range(len(cities_en_trois_d)):
        ret[i] = cities_en_trois_d[i][:2]
    return ret
#petit tests mignon pas du tout à sa place qui plot les villes et les nouveaux centroïdes, et colorie de la même couleur les clustered cities
def plot_kmeans(cities_coordinates, k, new_centroids):
    plt.scatter(cities_coordinates[:, 0], cities_coordinates[:, 1], c='black')
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red')
    plt.show()

def find_x(height=4, earth_radius=50):
    return np.sqrt(2 * height * earth_radius + height ** 2)

def I(r,coef=1) :
    return coef/r**2

def I_sph(sph_cities_coords, cities_coords,x_grid,y_grid,z_grid) :
    satellites_coords=np.array([x_grid,y_grid,z_grid]).T
    alpha_matrix=coef_sphere(sph_cities_coords, cities_coords, x_grid, y_grid, z_grid)
    distance_matrix=cdist(cities_coords, satellites_coords)
    I_return=np.zeros((np.shape(alpha_matrix)[0],np.shape(alpha_matrix)[1]))
    for i in range(np.shape(alpha_matrix)[0]) :
        for j in range(np.shape(alpha_matrix)[1]) :
            I_return[i,j]=I(distance_matrix[i,j])*alpha_matrix[i,j]
    return I_return

def I_tore(sph_cities_coords, cities_coords,x_grid,y_grid,z_grid) :
    satellites_coords=np.array([x_grid,y_grid,z_grid]).T
    alpha_matrix=coef_tore(sph_cities_coords, cities_coords, x_grid, y_grid, z_grid)
    distance_matrix=cdist(cities_coords, satellites_coords)
    I_return=np.zeros((np.shape(alpha_matrix)[0],np.shape(alpha_matrix)[1]))
    for i in range(np.shape(alpha_matrix)[0]) :
        for j in range(np.shape(alpha_matrix)[1]) :
            I_return[i,j]=I(distance_matrix[i,j])*alpha_matrix[i,j]
    return I_return

def coef_tore (sph_cities_coords, cities_coords, x_grid, y_grid, z_grid) :
    nrmls_cities=compute_torus_normals(sph_cities_coords[:,0],sph_cities_coords[:,1])
    satellites_coords=np.array([x_grid,y_grid,z_grid]).T
    delta_coords=satellites_coords-cities_coords[:,np.newaxis]
    alpha_matrix=np.sum(delta_coords*nrmls_cities[:,np.newaxis],axis=2)/np.linalg.norm(delta_coords,axis=2)
    alpha_matrix[alpha_matrix<0]=0
    return alpha_matrix

def coef_sphere(sph_cities_coords, cities_coords, x_grid, y_grid, z_grid) :
    nrmls_cities=compute_sph_normals(sph_cities_coords[:,0],sph_cities_coords[:,1])
    satellites_coords=np.array([x_grid,y_grid,z_grid]).T
    delta_coords=satellites_coords-cities_coords[:,np.newaxis]
    alpha_matrix=np.sum(delta_coords*nrmls_cities[:,np.newaxis],axis=2)/np.linalg.norm(delta_coords,axis=2)
    alpha_matrix[alpha_matrix<0]=0
    return alpha_matrix

def compute_torus_normals(theta, phi):
    """
    Compute normals to the surface of a torus at given theta and phi values.

    Parameters:
    - theta: Angle theta in radians (array)
    - phi: Angle phi in radians (array)

    Returns:
    - normals: Normals to the surface of the torus at given theta and phi values (array)
    """
    normal_x = np.cos(theta) * np.cos(phi)
    normal_y = np.sin(theta) * np.cos(phi)
    normal_z = np.sin(phi)
    return np.array([normal_x, normal_y, normal_z]).T

def compute_sph_normals(theta, phi) :
    normal_x=np.sin(phi)*np.cos(theta)
    normal_y=np.sin(phi)*np.sin(theta)
    normal_z=np.cos(phi)
    return np.array([normal_x,normal_y,normal_z]).T

def minimum_intensity(height, earth_radius, I) :
    thetamax=np.pi/2-np.arccos(earth_radius/(height+earth_radius))
    thetacool=thetamax/1.5
    b=2*np.sin(thetacool/2)*earth_radius
    alpha=(np.pi-thetacool)/2
    rangle=np.sqrt(height**2+b**2-2*height*b*np.cos(np.pi-alpha))
    Imin=I(rangle)
    return Imin,rangle

def minimum_intensity2(height,I) :
    Imax=I(height)
    return Imax/3

def pui_coef(point_satt, city_coordinates, height = 4, earth_radius=50) :
    delta = point_satt-city_coordinates
    norm_a = np.linalg.norm(city_coordinates)
    r = np.linalg.norm(delta)
    return max(0.0, np.dot(delta,city_coordinates)/norm_a/r)

def spherical_to_cartesian(spherical_coordinates, center, radius):
    x = center[0] + radius * np.sin(spherical_coordinates[:, 1]) * np.cos(
        spherical_coordinates[:, 0])
    y = center[1] + radius * np.sin(spherical_coordinates[:, 1]) * np.sin(
        spherical_coordinates[:, 0])
    z = center[2] + radius * np.cos(spherical_coordinates[:, 1])
    return np.c_[x, y, z]

def cartesian_to_spherical(cartesian_coordinates):
    x, y, z = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1], cartesian_coordinates[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.c_[theta, phi]

def spherical_to_cartesian_torus(spherical_coordinates, center, radius, cross_section_radius):
    x = center[0] + (radius + cross_section_radius * np.cos(spherical_coordinates[:, 1])) * np.cos(
        spherical_coordinates[:, 0])
    y = center[1] + (radius + cross_section_radius * np.cos(spherical_coordinates[:, 1])) * np.sin(
        spherical_coordinates[:, 0])
    z = center[2] + cross_section_radius * np.sin(spherical_coordinates[:, 1])
    return np.c_[x, y, z]
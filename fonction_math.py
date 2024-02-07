import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def euclildean_distance(city_coords, satellite_coords):
	"""
    Calculate the Euclidean distance between a city and a satellite in a 3D space.

    Parameters:
    - city_coords (numpy array): A tuple containing the x, y, and z coordinates of the city.
    - satellite_coords (numpy array): A tuple containing the x, y, and z coordinates of the satellite.

    Returns:
    float: The Euclidean distance between the city and the satellite.
	"""
	delta = city_coords-satellite_coords
	return math.sqrt(np.dot(delta,delta))

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
        sum_inverse_dist_squared += 1/euclildean_distance(city_coords, satellite_coord)**2
    return sum_inverse_dist_squared*satellite_power

def city_intensities(cities_coords, satellites_coords, satellite_power):
    intensities = np.zeros(np.shape(cities_coords)[0])
    for i, city_coord in enumerate(cities_coords):
        intensities[i] = total_intensity(city_coord, satellites_coords, satellite_power)
    return intensities

def is_covered(city_coords, satellites_coords):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        surface_radius = math.sqrt(satellite_coords[3]**2 - satellite_coords[2]**2) #Pythagora sqrt(r^2 - h^2)
        city_satellite_distance = math.sqrt((city_coords[0]-satellite_coords[0])**2 + (city_coords[1]-satellite_coords[1])**2)
        #print(surface_radius)
        #print(city_satellite_distance)
        if surface_radius >= city_satellite_distance:
            return True
    return False

def plot_covering(cities_coordinates, satellites_coordinates):
    
    # Extraire les coordonnées x et y des points
    x_coords, y_coords, z_coords = zip(*cities_coordinates)

    # Tracer les points
    for x_coord, y_coord in zip(x_coords, y_coords):
        plt.scatter(x_coord, y_coord, color='green' if is_covered(np.array([x_coord, y_coord]), satellites_coordinates) else 'red', marker='o')

    # Tracer les cercles
    for center, radius in zip(satellites_coordinates[:, :2], np.sqrt(satellites_coordinates[:, 3]**2 - satellites_coordinates[:, 2]**2)):
        circle = Circle(center, radius, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(circle)

    plt.title('Network coverage')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.axis('equal')
    
    plt.show()




import math
import numpy as np

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
    intensities = np.zeros(np.shape(cities_coords)[1])
    for i, city_coord in enumerate(cities_coords):
        intensities[i] = total_intensity(city_coord, satellites_coords, satellite_power)
    return intensities

def is_covered(city_coords, satellites_coords, satellite_radius):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        surface_radius = math.sqrt(satellite_radius**2 - satellite_coords[2]**2) #Pythagora sqrt(r^2 - h^2)
        city_satellite_distance = math.sqrt((city_coords[0]-satellite_coords[0])**2 + (city_coords[1]-satellite_coords[1])**2)
        print(surface_radius)
        print(city_satellite_distance)
        if surface_radius >= city_satellite_distance:
            return True
    return False


"""Quelques exemples"""

city_A = np.array([20, 10, 0])
city_B = np.array([10, 6, 0])
city_C = np.array([50, 50, 0])
city_D = np.array([40, 30, 0])
city_E = np.array([40, 10, 0])
satellite_A = np.array([20, 10, 30])
satellite_B = np.array([50, 40, 30])
satellite_C = np.array([70, 60, 30])
cities_coords = np.array([city_A, city_B, city_C])
satellites_coords = np.array([satellite_A, satellite_B, satellite_C])

print(euclildean_distance(city_A,satellite_A))
print(euclildean_distance(city_A,satellite_B))
print(euclildean_distance(city_A,satellite_C))

print(total_intensity(city_A, satellites_coords, 100))

print(city_intensities(cities_coords, satellites_coords, 100))

print(is_covered(city_B, satellites_coords, 32))
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

def total_intensity(city_coords, satellites_coords, satellite_intensity):
    """
    Calculate the total intensity at a city location based on an inverse square law.

    Parameters:
    - city_coords (numpy array): A tuple containing the x, y, and z coordinates of the city.
    - satellites_coords (numpy array): A list of numpy arrays, each containing the x, y, and z coordinates of a satellite.
    - satellite_intensity (float): Intensity of each satellite signal.

    Returns:
    float: The total intensity at the city location.
    """
    sum_inverse_dist_squared = 0
    for satellite_coord in satellites_coords:
        sum_inverse_dist_squared += 1/euclildean_distance(city_coords, satellite_coord)**2
    return sum_inverse_dist_squared*satellite_intensity

def city_intensities(cities_coords, satellites_coords, satellite_intensity):
    intensities = np.zeros(np.shape(cities_coords)[1])
    for i, city_coord in enumerate(cities_coords):
        intensities[i] = total_intensity(city_coord, satellites_coords, satellite_intensity)
    return intensities
      

"""Quelques exemples"""

city_A = np.array([0, 10, 20])
city_B = np.array([0, 20, 20])
city_C = np.array([0, 50, 100])
city_D = np.array([0, 30, 120])
city_E = np.array([0, 10, 10])
satellite_A = np.array([30, 30, 30])
satellite_B = np.array([50, 40, 30])
satellite_C = np.array([70, 60, 50])
cities_coords = np.array([city_A, city_B, city_C])
satellites_coords = np.array([satellite_A, satellite_B, satellite_C])

print(euclildean_distance(city_A,satellite_A))
print(euclildean_distance(city_A,satellite_B))
print(euclildean_distance(city_A,satellite_C))

print(total_intensity(city_A, satellites_coords, 100))


print(city_intensities(cities_coords, satellites_coords, 100))


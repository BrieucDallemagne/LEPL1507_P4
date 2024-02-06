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
	sum_inverse_dist_squared = 0
	for satellite_coord in satellites_coords:
		sum_inverse_dist_squared += 1/euclildean_distance(city_coords, satellite_coord)**2
	return sum_inverse_dist_squared*satellite_intensity

"""Quelques exemples"""

city_A = np.array([0, 10, 20])
satellite_A = np.array([30, 30, 30])
satellite_B = np.array([50, 40, 30])
satellite_C = np.array([70, 60, 50])
print(euclildean_distance(city_A,satellite_A))
print(euclildean_distance(city_A,satellite_B))
print(euclildean_distance(city_A,satellite_C))

satellites_coords = np.array([satellite_A, satellite_B, satellite_C])
print(total_intensity(city_A, satellites_coords, 100))

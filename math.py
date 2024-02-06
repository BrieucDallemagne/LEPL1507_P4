import math

def euclildean_distance(city_coords,satellite_coords):
	"""
    Calculate the Euclidean distance between a city and a satellite in a 3D space.

    Parameters:
    - city_coords (array): A tuple containing the x, y, and z coordinates of the city.
    - satellite_coords (array): A tuple containing the x, y, and z coordinates of the satellite.

    Returns:
    float: The Euclidean distance between the city and the satellite.
	"""
	return math.sqrt((city_coords[0]-satellite_coords[0])**2 + (city_coords[1]-satellite_coords[1])**2 + (city_coords[2]-satellite_coords[2])**2)

def 

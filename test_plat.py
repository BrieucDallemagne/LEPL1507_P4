import euclidean_satellites_repartition as esr
import numpy as np
import matplotlib.pyplot as plt
import fonction_math as fm
import plot_plat as pp

"""Quelques exemples"""
city_A = np.array([1, 2, 0]) # x, y, z
city_B = np.array([4, 8, 0])
city_C = np.array([5, 2, 0])
city_D = np.array([8, 14, 0])
city_E = np.array([10, 6, 0])
satellite_A = np.array([2, 5, 12, 13]) # x, y, z, radius
satellite_B = np.array([4, 3, 12, 13])
satellite_C = np.array([11, 8, 12, 12.5])
cities_coordinates = np.array([city_A, city_B, city_C, city_D, city_E])
satellites_coordinates = np.array([satellite_A, satellite_B, satellite_C])

"""print(euclildean_distance(city_A, satellite_A[:3]))
print(euclildean_distance(city_A, satellite_B[:3]))
print(euclildean_distance(city_A, satellite_C[:3]))

print(total_intensity(city_A, satellites_coordinates[:, :3], 100))

print(city_intensities(cities_coordinates, satellites_coordinates[:, :3], 100))
"""

print(fm.is_covered(city_A, satellites_coordinates))
print(fm.is_covered(city_B, satellites_coordinates))
print(fm.is_covered(city_C, satellites_coordinates))
print(fm.is_covered(city_D, satellites_coordinates))
print(fm.is_covered(city_E, satellites_coordinates))

fm.plot_covering(cities_coordinates, satellites_coordinates)

"""fin exemple"""


#début exemple
# Définir les coordonnées des villes
cities = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])

# Définir les coordonnées des satellites
satellites = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Définir les intensités des satellites
intensities = np.array([1, 1, 1, 1])

# Calculer les intensités des villes
intensities_cities = np.zeros(cities.shape[0])

for i, city in enumerate(cities):
    for j, satellite in enumerate(satellites):
        intensities_cities[i] += 1/np.sqrt(np.sum((city-satellite)**2))*intensities[j]


pp.plot_plat(cities, satellites)

#fin exemple

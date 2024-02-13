import euclidean_satellites_repartition as esr
import numpy as np
import matplotlib.pyplot as plt
import fonction_math as fm
import plot_plat as pp
import random

"""Quelques exemples
city_A = [1, 2] # x, y
city_B = [4, 8]
city_C = [5, 2]
city_D = [8, 14]
city_E = [10, 6]
#satellite_A = np.array([2, 5, 12, 13]) # x, y, z, radius
#satellite_B = np.array([4, 3, 12, 13])
#satellite_C = np.array([11, 8, 12, 12.5])
europe = np.array([city_A, city_B, city_C, city_D, city_E])

city_F = [0, 1]
city_G = [0, 2]
city_H = [0, 5]
city_I = [0, 9]
city_J = [0, 10]

asia = np.array([city_F, city_G, city_H, city_I, city_J])

planet = [europe, asia]
print(euclildean_distance(city_A, satellite_A[:3]))
print(euclildean_distance(city_A, satellite_B[:3]))
print(euclildean_distance(city_A, satellite_C[:3]))

print(total_intensity(city_A, satellites_coordinates[:, :3], 100))

print(city_intensities(cities_coordinates, satellites_coordinates[:, :3], 100))


radius_list = [1.0, 3.0, 5.0]

for continent in planet:
    for radius in radius_list:

        # Résoudre le problème pour le rayon actuel
        satellites_coordinates = fm.solve_2D(continent, 15, radius)

        # Afficher les résultats
        print("Satellite coordinates:")
        print(satellites_coordinates)

        for city in continent:
            print(fm.is_covered_2D(city, satellites_coordinates))

        # Tracer la couverture du réseau
        print("\nPlotting coverage:")
        fm.plot_covering_2D(continent, satellites_coordinates, 15)



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
"""
def test_solve_2D_random(grid_size=10, radius=1.0, n_cities=10, n_tests=10, number_of_satellites=10):
    for i in range(n_tests):
        radius = random.uniform(0.1, 3.0)
        radius = round(radius, 2)
        n_cities = random.randint(1, 20)
        poids = fm.create_weight(n_cities)

        cities_coordinates = np.random.randint(0, grid_size, size=(n_cities, 2))
        nbr_max_sat = fm.nbr_max_sat(cities_coordinates, grid_size, radius)
        number_of_satellites = random.randint(1, nbr_max_sat)
        satellites_coordinates = fm.solve_2D(number_of_satellites,cities_coordinates, poids, grid_size, radius )
        fm.plot_covering_2D(cities_coordinates, satellites_coordinates, grid_size)
        plt.show()


test_solve_2D_random()
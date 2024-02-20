import euclidean_satellites_repartition as esr
import numpy as np
import matplotlib.pyplot as plt
import fonction_math as fm
import plot_plat as pp
import random


def test_solve_2D_random(grid_size=100, n_cities=10, scope=1000, height=100, max_sat=9):
    poids = fm.create_weight(n_cities)
    radius = np.sqrt(scope**2 - height**2)

    cities_coordinates = np.random.randint(0, grid_size, size=(n_cities, 2))
    number_of_satellites = max_sat
    #print('nombres de satellites optimal :', nbr_max_sat)
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, poids, grid_size, scope, height, intensity=1000)
    #fm.plot_covering_2D(cities_coordinates, poids, satellites_coordinates, grid_size)
    #plt.show()




def test_special_cases():
    # Test de la fonction solve_2D pour des cas particuliers
    # Cas 1: limites de la version discrete
    cities_coordinates = np.array([[0, 0],[0,1]])
    radius = 0.95
    number_of_satellites = 1
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [0.8,0.2], 2, scope = 1, height = 0.05)
    fm.plot_covering_2D(cities_coordinates, [0.8,0.2], satellites_coordinates, 2)

    # cas 2 : 2 villes au même endroit
    cities_coordinates = np.array([[0, 0],[0,0]])
    radius = 1
    number_of_satellites = 1
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [0.5,0.5], 2, scope=1, height=0)
    fm.plot_covering_2D(cities_coordinates, [0.5,0.5], satellites_coordinates, 2)

    # cas 3 : 1 satellite have to choice between 2 cities of big intensity and 3 cities of small intensity
    cities_coordinates = np.array([[0, 0],[0,1],[1,0],[3,3],[3,4]])
    radius = 2
    number_of_satellites = 1
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [0.1,0.01,0.09,0.4,0.4], 5, scope = 2, height = 0)
    fm.plot_covering_2D(cities_coordinates, [0.1,0.01,0.09,0.4,0.4], satellites_coordinates, 5)

    # cas 4 : find best position between 4 satellites
    cities_coordinates = np.array([[1, 2],[0,1],[1,0],[2,1]])
    radius = 1
    number_of_satellites = 1
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [0.25,0.25,0.25,0.25], 5, scope = 1, height = 0)
    fm.plot_covering_2D(cities_coordinates, [0.25,0.25,0.25,0.25], satellites_coordinates, 5)

    # cas 5 : as many satellites as cities
    cities_coordinates = np.array([[1, 2],[0,1],[1,0],[2,1]])
    radius = 0.1
    number_of_satellites = 4
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [0.25,0.25,0.25,0.25], 5, scope = 1 , height = 0.7)
    fm.plot_covering_2D(cities_coordinates, [0.25,0.25,0.25,0.25], satellites_coordinates, 5)

    # cas 6 : 2 satellites for 1 city
    cities_coordinates = np.array([[1, 2]])
    radius = 1
    number_of_satellites = 2
    satellites_coordinates = fm.solve_2D(number_of_satellites, cities_coordinates, [1], 5, scope = 1, height = 0)
    fm.plot_covering_2D(cities_coordinates, [1], satellites_coordinates, 5)



#test_solve_2D_random()

#test_special_cases()




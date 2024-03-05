import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import plot_rond as pr
import math
import matplotlib
matplotlib.use('TkAgg')
def test_solve_3D_random(k_means=False):
    n_cities = 5
    cities_weights = fm.create_weight(n_cities)
def test_solve_3D_random(n_tests=5, k_means=False):
    for i in range(n_tests):
        n_cities = np.random.randint(2, 20)
        #weights = fm.create_weight(n_cities)
        weights = np.full(n_cities, 1/n_cities)
        new_count = 0
        cities_weights = np.array([])
        for weight in range(n_cities):
            if weights[weight] > 1/(n_cities*100):
                cities_weights = np.append(cities_weights, weights[weight])
                new_count += 1
        
        n_cities = new_count 
        radius_earth = 50
        cities_coordinates_latitude = np.random.randint(-90, 90, size=(n_cities))
        cities_coordinates_longitude = np.random.randint(-180, 180, size=(n_cities))
        cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]

        cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
        cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
        cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
        cities_coordinates = np.c_[cities_x, cities_y, cities_z]
        original_cities = cities_coordinates
        if k_means:
            cities_coordinates, poids = fm.k_means_cities(cities_coordinates, n_cities-1, poids)
        #print(cities_coordinates)
        number_of_satellites = np.random.randint(1, n_cities)
        satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights,  10)
        if np.array_equal(satellites_coordinates, np.array([])):
            continue
        if k_means:
            pr.plot_3D(cities_coordinates, satellites_coordinates,  10, True, original_cities)
        else:
            pr.plot_3D(cities_coordinates, satellites_coordinates,  10)
        plt.show()

test_solve_3D_random( n_tests=5, k_means=False)
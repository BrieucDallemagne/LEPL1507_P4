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

def test_solve_3D_random(n_tests=5, k_means=False, fix_seed = False, verbose = False, planet=False):
    for i in range(n_tests):
        if fix_seed:
            cities_weights = [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            radius_earth = 50
            cities_coordinates_latitude = [0, 45, -45, 0, 80, 12, -26, -60, -15]
            cities_coordinates_longitude = [0, 30, 90, 120, 85, -157, -76, -34, 170]
            n_cities = len(cities_coordinates_latitude)
        else:
            n_cities = np.random.randint(5, 100)
            #cities_weights = fm.create_weight(n_cities)
            cities_weights = np.full(n_cities, 1/n_cities)
            radius_earth = 50

            cities_coordinates_latitude = np.random.randint(-90, 90, size=(n_cities))
            cities_coordinates_longitude = np.random.randint(-180, 180, size=(n_cities))
        og_cit = np.array(0)
        og_weights = np.array(0)
        if k_means:
            print("hey")
            cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
            cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in
                        cities_coordinates]
            cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in
                        cities_coordinates]
            cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
            og_cit = np.c_[cities_x, cities_y, cities_z]
            og_weights = cities_weights
            cities_long_and_lat = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
            cities_wrap = fm.k_means_cities(cities_long_and_lat, n_cities//2, cities_weights)
            cities_long_and_lat = cities_wrap[0]
            cities_weights = cities_wrap[1]
            cities_coordinates_latitude = cities_long_and_lat[:, 0]
            cities_coordinates_longitude = cities_long_and_lat[:, 1]
        cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]

        cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
        cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
        cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
        cities_coordinates = np.c_[cities_x, cities_y, cities_z]
        satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
        if np.array_equal(satellites_coordinates, np.array([])):
            continue
        if k_means:
            pr.plot_3D(og_cit, satellites_coordinates, og_weights, 10, k_means,  centroids = cities_coordinates, centroids_weights = cities_weights, rot=False, planet = "earth")
        else:
            pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, False, centroids=np.array(0),
                   centroids_weights=np.array(0), rot=False, planet="earth")
        plt.show()


test_solve_3D_random(n_tests=1, k_means=False, fix_seed=False, verbose=False)
test_solve_3D_random(n_tests=1, k_means=True, fix_seed=False, verbose=False)
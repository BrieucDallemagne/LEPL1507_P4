import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import plot_rond as pr

def test_solve_3D_random( n_tests=10):
    for i in range(n_tests):
        n_cities = np.random.randint(1, 20)
        poids = fm.create_weight(n_cities)
        cities_coordinates_latitude = np.random.randint(-90, 90, size=(n_cities))
        cities_coordinates_longitude = np.random.randint(-180, 180, size=(n_cities))
        cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
        #number_of_satellites = np.random.randint(1, n_cities)
        #satellites_coordinates = ssr.spherical_satellites_repartition(number_of_satellites, cities_coordinates, poids)
        satellites_coordinates = [(10000, 0, 0), (0, 10000, 0)]  # Example satellite coordinates (x, y, z)
        pr.plot_3D(satellites_coordinates, cities_coordinates)
        plt.show()


test_solve_3D_random( n_tests=10)
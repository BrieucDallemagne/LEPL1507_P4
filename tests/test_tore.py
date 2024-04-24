import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tore_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import plot_tore as pr
import math
import matplotlib
import csv

matplotlib.use('TkAgg')

def test_solve_3D_random(n_tests=5, k_means=False, real_cities = False, verbose = False, planet=False):
    for i in range(n_tests):
        if real_cities:
            n_cities = np.random.randint(4,5)
            file = open('worldcities.csv', 'r', encoding='utf-8')
            csv_reader = csv.DictReader(file)

            longitudes = []
            latitudes = []
            populations = []
            cities_names = []
            countries_names = []
            
            num_rows = sum(1 for row in csv_reader)
            file.seek(0)
            random_indices = random.sample(range(num_rows), n_cities)

            for i, row in enumerate(csv_reader):
                if i in random_indices:
                    longitudes.append(float(row['lng']))
                    latitudes.append(float(row['lat']))
                    populations.append(row['population'])
                    cities_names.append(row['city'])
                    countries_names.append(row['country'])

            cities_coordinates_sph = np.array([longitudes, latitudes]).T
            cities_coordinates_sph[:, 1] = (180 - 2*cities_coordinates_sph[:, 1]) 
            cities_coordinates = np.radians(cities_coordinates_sph)
            print("Villes affichées :")
            for i in range(n_cities):
                print("{}, {}".format(cities_names[i], countries_names[i]))


            cities_weights = np.full(cities_coordinates_sph.shape[0], 1/cities_coordinates_sph.shape[0])
        else:
            n_cities = 40
            #cities_weights = fm.create_weight(n_cities)
            cities_weights = np.full(n_cities, 1/n_cities)
            radius_earth = 50
            
            cities_coordinates_latitude = np.radians(np.random.randint(-180, 180, size=(n_cities)))
            cities_coordinates_longitude = np.radians(np.random.randint(-180, 180, size=(n_cities)))
            cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
        
        og_cit = np.array(0)
        og_weights = np.array(0)
        if k_means:
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
        satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
        if np.array_equal(satellites_coordinates, np.array([])):
            continue
        if k_means:
            
            pr.plot_torus(og_cit, satellites_coordinates, og_weights, 10, k_means,  centroids = cities_coordinates, centroids_weights = cities_weights, rot=False, planet = "earth")
        else:
            
            pr.plot_torus(cities_coordinates, satellites_coordinates, cities_weights, 10, False, centroids=np.array(0),
                   centroids_weights=np.array(0), rot=False, planet="earth")
        plt.show()


test_solve_3D_random(n_tests=1, k_means=False, real_cities=False, verbose=False)


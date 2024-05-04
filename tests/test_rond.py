import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import src.fonction_math as fm
import plots.plot_rond as pr
import math
import matplotlib
import csv

matplotlib.use('TkAgg')

def test_solve_3D_random(n_cities=100,n_tests=5, k_means=False, real_cities = False, verbose = False, planet=False):
    for i in range(n_tests):
        if real_cities:
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
            cities_coordinates_sph[:, 1] = (90 - cities_coordinates_sph[:, 1]) 
            cities_coordinates = np.radians(cities_coordinates_sph)
            cities_weights = np.full(cities_coordinates_sph.shape[0], 1/cities_coordinates_sph.shape[0])

            if k_means:
                centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, cities_coordinates_sph.shape[0]//2, cities_weights, spherical=True)
                satellites_coordinates = ssr.spherical_satellites_repartition(fm.cartesian_to_spherical(centroids_coordinates), centroids_weights, 10, verbose=verbose)
            else:
                satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
            
            if k_means:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=True, centroids=centroids_coordinates)
            else:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False)
            plt.show()

        else:
            n_cities = np.random.randint(5, 100)
            cities_weights = np.full(n_cities, 1/n_cities)

            cities_coordinates_latitude = np.radians(np.random.randint(-90, 90, size=(n_cities)))
            cities_coordinates_longitude = np.radians(np.random.randint(-180, 180, size=(n_cities)))
            cities_coordinates = np.c_[cities_coordinates_longitude, cities_coordinates_latitude]

            if k_means:
                centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, cities_coordinates_sph.shape[0]//2, cities_weights, spherical=True)
                satellites_coordinates = ssr.spherical_satellites_repartition(fm.cartesian_to_spherical(centroids_coordinates), centroids_weights, 10, verbose=verbose)
            else:
                satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
            
            if k_means:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=True, centroids=centroids_coordinates)
            else:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False)
            plt.show()
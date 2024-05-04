import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import src.spherical_satellites_repartition as ssr
import src.euclidean_satellites_repartition as esr
import pandas as pd
import tests.test_rond as tr
import pandas as pd
import plots.plot_plat as pp
import plots.plot_rond as pr
import src.fonction_math as fm

def resolve_rond(n_cities,csv_name, kmeans=False,verbose=False):
            
            df = pd.read_csv(csv_name)
            sizes = df["size"].values
            cities_names = df["villeID"].values
            longitudes = df["long"].values
            latitudes = df["lat"].values
                    
            cities_coordinates_sph = np.array([longitudes, latitudes]).T
            cities_coordinates_sph[:, 1] = (90 - cities_coordinates_sph[:, 1]) 
            cities_coordinates = np.radians(cities_coordinates_sph)
            cities_weights = [float(i)/sum(sizes) for i in sizes]

            if kmeans:
                centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, n_cities//2, cities_weights, spherical=True)
                satellites_coordinates = ssr.spherical_satellites_repartition(fm.cartesian_to_spherical(centroids_coordinates), centroids_weights, 10, verbose=verbose)
            else:
                satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
            
            if kmeans:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=True, centroids=centroids_coordinates)
            else:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False)
            plt.show()

def resolve_carre(n_cities,csv_name,kmeans=False,verbose=False):
                
    df = pd.read_csv(csv_name)
    sizes = df["size"].values
    cities_names = df["villeID"].values
    X = df["Xi"].values
    Y = df["Yi"].values
    cities_coordinates = np.c_[X, Y]
    cities_weights = [float(i)/sum(sizes) for i in sizes]
    height = np.random.randint(1, 10)
    scope = np.random.randint(height, 100)
    grid_size = 400

    number_of_satellites = np.random.randint(1, n_cities)
    if kmeans:
        centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, n_cities // 2,
                                                                    sizes)
        satellites_coordinates = esr.euclidean_satellites_repartition(number_of_satellites, centroids_coordinates,
                                                                    centroids_weights, grid_size, scope, height, verbose=verbose)
        pp.plot_covering_2D(cities_coordinates, sizes, satellites_coordinates, centroids = centroids_coordinates)
    else:
        satellites_coordinates = esr.euclidean_satellites_repartition(number_of_satellites, cities_coordinates,
                                                                    sizes, grid_size, scope, height, verbose=verbose)
        pp.plot_covering_2D(cities_coordinates, sizes, satellites_coordinates)
    
    plt.show()
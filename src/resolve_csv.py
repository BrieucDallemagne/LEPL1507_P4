import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import src.spherical_satellites_repartition as ssr
import src.euclidean_satellites_repartition as esr
import pandas as pd
import test.test_rond as tr
import pandas as pd
import plots.plot_plat as pp
import plots.plot_rond as pr
import src.fonction_math as fm

def resolve_rond(n_cities,csv_name):
            
            df = pd.read_csv(csv_name)
            sizes = df["size"].values
            cities_names = df["villeID"].values
            longitudes = df["long"].values
            latitudes = df["lat"].values
                    
            cities_coordinates_sph = np.array([longitudes, latitudes]).T
            cities_coordinates_sph[:, 1] = (90 - cities_coordinates_sph[:, 1]) 
            cities_coordinates = np.radians(cities_coordinates_sph)
            #print("Villes affichées :")
            #for i in range(n_cities):
                #print("{}, {}".format(cities_names[i], countries_names[i]))

            #normalize the weights
            cities_weights = [float(i)/sum(sizes) for i in sizes]

            satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=False)
            pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False, planet="earth")
            plt.show()

def resolve_carre(n_cities,csv_name):
                
    df = pd.read_csv(csv_name)
    sizes = df["size"].values
    cities_names = df["villeID"].values
    X = df["Xi"].values
    Y = df["Yi"].values
    cities_coordinates = np.c_[X, Y]
    cities_weights = [float(i)/sum(sizes) for i in sizes]

    number_of_satellites = np.random.randint(1, n_cities)
    satellites_coordinates = esr.euclidean_satellites_répartition(number_of_satellites, cities_coordinates, cities_weights, 180, 200, 100, intensity=100000)
    pp.plot_covering_2D(cities_coordinates, cities_weights, satellites_coordinates, 100)
    plt.show()

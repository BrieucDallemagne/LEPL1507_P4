import matplotlib.pyplot as plt
import src.fonction_math as fm
import numpy as np
import pyvista as pv
from plots.plot_rond import plot_3D
def plot_centroids_and_cities(cities_coordinates, cities_weights, new_centroids, centroids_weight):
    fig, ax = plt.subplots()
    ax.scatter(cities_coordinates[:, 0], cities_coordinates[:, 1], c='blue', label='Cities')
    ax.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', label='Centroids')
    for i, txt in enumerate(cities_weights):
        ax.annotate(txt, (cities_coordinates[i, 0], cities_coordinates[i, 1]))
    for i, txt in enumerate(centroids_weight):
        ax.annotate(txt, (new_centroids[i, 0], new_centroids[i, 1]))
    plt.legend()
    plt.show()

cities_c = np.array([[1, 2],[0,1],[1,0],[2,1]])
cities_w = np.array([0.25,0.25,0.25,0.25])
new_centroids = fm.k_means_cities(cities_c, 2, cities_w)
plot_centroids_and_cities(cities_c, cities_w, new_centroids[0], new_centroids[1])


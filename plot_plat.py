import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

#fin exemple


def plot_plat(coord_cities, coord_satellites):
    """
    Plot the cities and the satellites on a 2D plane.

    Parameters:
    - coord_cities (numpy array): A list of numpy arrays, each containing the x, y, and z coordinates of a city.
    - coord_satellites (numpy array): A list of numpy arrays, each containing the x, y, and z coordinates of a satellite.

    Returns:
    None
    """
    # Définir les coordonnées du plan
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coord_cities[:,0], coord_cities[:,1],coord_cities[:,2], c='r', marker='o', label='Cities')
    ax.scatter(coord_satellites[:,0], coord_satellites[:,1],coord_satellites[:,2], c='b', marker='^', label='Satellites')
    ax.plot_surface(X, Y, Z, color='g', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

plot_plat(cities, satellites)






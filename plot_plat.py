import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    ax.set_zlabel('hauteur')
    print('hauteur des satellites :', coord_satellites[0,2])
    plt.show()








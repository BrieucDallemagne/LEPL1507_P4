import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm

def plot_3D(satellites_coordinates, cities_coordinates):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    radius_earth = 6371  
    latitudes = np.linspace(-90, 90, 100)  
    longitudes = np.linspace(-180, 180, 200)  
    longitude_grid, latitude_grid = np.meshgrid(np.radians(longitudes), np.radians(latitudes))
    x = radius_earth * np.cos(latitude_grid) * np.cos(longitude_grid)
    y = radius_earth * np.cos(latitude_grid) * np.sin(longitude_grid)
    z = radius_earth * np.sin(latitude_grid)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.1, linewidth=0)

    # Plot satellites
    satellites_x = [coord[0] for coord in satellites_coordinates]
    satellites_y = [coord[1] for coord in satellites_coordinates]
    satellites_z = [coord[2] for coord in satellites_coordinates]
    ax.scatter(satellites_x, satellites_y, satellites_z, color='r', marker='o', label='Satellites')

    # Plot cities
    cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
    cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
    cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
    ax.scatter(cities_x, cities_y, cities_z, color='g', marker='^', label='Cities')

    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth Model with Satellites and Cities')
    ax.legend()

    plt.show()




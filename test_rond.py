import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3D():
    # Constants
    radius_earth = 6371  # Earth's radius in kilometers

    # Generate latitude and longitude values
    latitudes = np.linspace(-90, 90, 100)  # Range of latitudes from -90 to 90
    longitudes = np.linspace(-180, 180, 200)  # Range of longitudes from -180 to 180

    # Create a meshgrid of latitudes and longitudes
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

    # Convert latitude and longitude to 3D Cartesian coordinates
    x = radius_earth * np.cos(np.radians(latitude_grid)) * np.cos(np.radians(longitude_grid))
    y = radius_earth * np.cos(np.radians(latitude_grid)) * np.sin(np.radians(longitude_grid))
    z = radius_earth * np.sin(np.radians(latitude_grid))

    # Plot Earth
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface of the Earth
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.6, linewidth=0)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth Model')

    # Show plot
    plt.show()


plot_3D()

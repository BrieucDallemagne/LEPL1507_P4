import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import math

def plot_3D_old(satellites_coordinates, cities_coordinates, kmeans=False, original_cities_coordinates=np.array(0)):
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
    if kmeans:
        ax.scatter(cities_x, cities_y, cities_z, color='g', marker='^', label='Centroïds')
        original_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
        original_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
        original_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in original_cities_coordinates]
        ax.scatter(original_x, original_y, original_z, color='y', marker='^', label='Cities')
    else:
        ax.scatter(cities_x, cities_y, cities_z, color='g', marker='^', label='Cities')

    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth Model with Satellites and Cities')
    ax.legend()

    plt.show()

def is_covered_3D(city_coords, satellites_coords, scope):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        city_satellite_distance = math.sqrt((city_coords[0]-satellite_coords[0])**2 + (city_coords[1]-satellite_coords[1])**2 + (city_coords[2]-satellite_coords[2])**2)
        if scope >= city_satellite_distance:
            return True
    return False


def plot_3D(cities_coordinates, satellites_coordinates, scope, height):

    sphere_center = (0, 0, 0)

    # Créer le plot en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rayon de la sphère
    earth_radius = 50
    satellite_radius = 50 + height

    # Créer la terre
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = sphere_center[0] + earth_radius * np.outer(np.cos(u), np.sin(v))
    y = sphere_center[1] + earth_radius * np.outer(np.sin(u), np.sin(v))
    z = sphere_center[2] + earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Dessiner la sphère
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    # Dessiner le quadrillage
    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    theta, phi = np.meshgrid(theta_values, phi_values)
    x_grid = sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)
    y_grid = sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)
    z_grid = sphere_center[2] + satellite_radius * np.cos(phi)

    #ax.plot_wireframe(x_grid, y_grid, z_grid, color='black', linewidth=0.5)

    #ax.scatter(x_grid, y_grid, z_grid, color='red', s=10, alpha=0.2)

    x_sat = sphere_center[0] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.cos(satellites_coordinates[:, 0])
    y_sat = sphere_center[1] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.sin(satellites_coordinates[:, 0])
    z_sat = sphere_center[2] + satellite_radius * np.cos(satellites_coordinates[:, 1])
    print("lol")
    print(satellites_coordinates)
    print(x_sat)
    print(y_sat)
    print(z_sat)
    satellites_spherical_coordinates = np.c_[x_sat, y_sat, z_sat]
    print(satellites_spherical_coordinates)

    ax.scatter(x_sat, y_sat, z_sat, color='blue', s=40, label='Satellites')
    for x_s, y_s, z_s in zip(x_sat, y_sat, z_sat):
        x = x_s + scope * np.outer(np.cos(u), np.sin(v))
        y = y_s + scope * np.outer(np.sin(u), np.sin(v))
        z = z_s + scope * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    
    for x_city, y_city, z_city in cities_coordinates:
        is_covered = is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
        print("City ({}, {}, {}) is covered: {}".format(x_city, y_city, z_city, is_covered))
        ax.scatter(x_city, y_city, z_city, c='green' if is_covered else "red", s=20, marker='o')

    # Configurer les limites de l'axe
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_zlim(-70, 70)

    # Make all axes equal in size
    ax.set_box_aspect([1, 1, 1])

    # Ajouter une légende
    ax.legend()

    # Afficher le plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
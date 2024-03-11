import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import math
from matplotlib import animation

def is_covered_3D(city_coords, satellites_coords, scope):
    """
    Check whether or not the city is covered by at least one satellite
    """
    for satellite_coords in satellites_coords:
        city_satellite_distance = math.sqrt(
            (city_coords[0] - satellite_coords[0]) ** 2 + (city_coords[1] - satellite_coords[1]) ** 2 + (
                        city_coords[2] - satellite_coords[2]) ** 2)
        if scope >= city_satellite_distance:
            return True
    return False


def plot_3D(cities_coordinates, satellites_coordinates, cities_weights, height, original_cities_coordinates=np.array(0), original_cities_weights=np.array(0), kmeans=False,rot=False):
    sphere_center = (0, 0, 0)

    # Créer le plot en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rayon de la sphère
    earth_radius = 50
    satellite_radius = 50 + height
    scope = fm.find_x(height, earth_radius)

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

    # ax.plot_wireframe(x_grid, y_grid, z_grid, color='black', linewidth=0.5)

    # ax.scatter(x_grid, y_grid, z_grid, color='red', s=10, alpha=0.2)

    x_sat = sphere_center[0] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.cos(
        satellites_coordinates[:, 0])
    y_sat = sphere_center[1] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.sin(
        satellites_coordinates[:, 0])
    z_sat = sphere_center[2] + satellite_radius * np.cos(satellites_coordinates[:, 1])

    # print(satellites_coordinates)
    # print(x_sat)
    # print(y_sat)
    # print(z_sat)
    satellites_spherical_coordinates = np.c_[x_sat, y_sat, z_sat]
    # print(satellites_spherical_coordinates)

    ax.scatter(x_sat, y_sat, z_sat, color='blue', s=40, label='Satellites')
    for x_s, y_s, z_s in zip(x_sat, y_sat, z_sat):
        x = x_s + scope * np.outer(np.cos(u), np.sin(v))
        y = y_s + scope * np.outer(np.sin(u), np.sin(v))
        z = z_s + scope * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    for x_city, y_city, z_city in cities_coordinates:
        is_covered = is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
        # print("City ({}, {}, {}) is covered: {}".format(x_city, y_city, z_city, is_covered))
        ax.scatter(x_city, y_city, z_city, c='green' if is_covered else "red", s=20, marker='o')
    i = 0
    if kmeans:
        for x_city, y_city, z_city in original_cities_coordinates:
            is_covered = is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
            # print("City ({}, {}, {}) is covered: {}".format(x_city, y_city, z_city, is_covered))
            ax.scatter(x_city, y_city, z_city, c='pink' if is_covered else "orange", s=20, marker='o')
            ax.text(x_city, y_city, z_city, '%s' % (str(original_cities_weights[i])), size=5, zorder=1,
                    color='k')
            i += 1
            # Configurer les limites de l'axe
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
    if rot:
        def animate(frame):
            ax.view_init(30, 6 * frame)
            plt.pause(.001)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=200, interval=100)
    plt.show()
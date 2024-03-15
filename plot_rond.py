import spherical_satellites_repartition as ssr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fonction_math as fm
import math
from matplotlib import animation
import pyvista as pv

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

def has_enough_intensity(city_coords, satellites_coords, min_intensity, scope):
    total_intensity = 0
    for satellite_coords in satellites_coords:
        city_satellite_distance = math.sqrt(
            (city_coords[0] - satellite_coords[0]) ** 2 + (city_coords[1] - satellite_coords[1]) ** 2 + (city_coords[2] - satellite_coords[2]) ** 2)
        if city_satellite_distance <= scope:
            total_intensity += fm.I(city_satellite_distance)
    return total_intensity >= min_intensity


def plot_3D_old(cities_coordinates, satellites_coordinates, cities_weights, height, original_cities_coordinates=np.array(0), original_cities_weights=np.array(0), kmeans=False,rot=False):
    sphere_center = (0, 0, 0)
    earth_radius = 50

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

    """# Dessiner le quadrillage
    theta_values = np.linspace(0, 2 * np.pi, 20)[1:]
    phi_values = np.linspace(0, np.pi, 20)[1:-1]

    theta, phi = np.meshgrid(theta_values, phi_values)
    x_grid = sphere_center[0] + satellite_radius * np.sin(phi) * np.cos(theta)
    y_grid = sphere_center[1] + satellite_radius * np.sin(phi) * np.sin(theta)
    z_grid = sphere_center[2] + satellite_radius * np.cos(phi)

    ax.plot_wireframe(x_grid, y_grid, z_grid, color='black', linewidth=0.5)

    ax.scatter(x_grid, y_grid, z_grid, color='red', s=10, alpha=0.2)"""

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
        ax.scatter(x_city, y_city, z_city, c='green' if is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.inten_min(height, earth_radius, fm.I)[0], scope) else
                                            "orange" if is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and not has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.inten_min(height, earth_radius, fm.I)[0], scope) else
                                            "red", s=20, marker='o')
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

def plot_3D(cities_coordinates, satellites_coordinates, cities_weights, height, original_cities_coordinates=np.array(0), original_cities_weights=np.array(0), kmeans=False,rot=False):
    #met le background en blanc
    sphere_center = (0, 0, 0)
    earth_radius = 50

    # Créer le plot en 3D
    plotter = pv.Plotter()
    plotter.set_background('white')
    earth_mesh = pv.examples.planets.load_earth()
    earth_mesh.points *= earth_radius
    texture = pv.read_texture("Planet_Images\earth.jpg")
    plotter.add_mesh(earth_mesh, texture=texture)

    # Autre planète (facultatif)
    second_mesh = pv.examples.planets.load_earth()
    second_texture = pv.read_texture("Planet_Images\sun.jpg")
    second_mesh.points = 300*second_mesh.points + np.array([1000, 1000, 1000])
    second_mesh.rotate_x(135)
    second_mesh.rotate_y(45)
    second_mesh.rotate_z(90)
    plotter.add_mesh(second_mesh, texture=second_texture)

    # Rayon de la sphère
    satellite_radius = 50 + height
    scope = fm.find_x(height, earth_radius)

    # Dessiner les satellites
    x_sat = sphere_center[0] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.cos(
        satellites_coordinates[:, 0])
    y_sat = sphere_center[1] + satellite_radius * np.sin(satellites_coordinates[:, 1]) * np.sin(
        satellites_coordinates[:, 0])
    z_sat = sphere_center[2] + satellite_radius * np.cos(satellites_coordinates[:, 1])

    for x_s, y_s, z_s in zip(x_sat, y_sat, z_sat):
        sphere = pv.Sphere(radius=scope, center=(x_s, y_s, z_s))
        plotter.add_mesh(sphere, color='firebrick', point_size=10, opacity=0.3)

    satellites_spherical_coordinates = np.c_[x_sat, y_sat, z_sat]

    # Dessiner les villes
    satisfied_proportion = 0
    for i, (x_city, y_city, z_city) in enumerate(cities_coordinates):
        is_covered = is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope)
        if has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.minimum_intensity(height, earth_radius, fm.I)[0], scope): satisfied_proportion += cities_weights[i]
        color = 'green' if is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.minimum_intensity(height, earth_radius, fm.I)[0], scope) else \
                'orange' if is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and not has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.minimum_intensity(height, earth_radius, fm.I)[0], scope) else \
                'red'
        if (is_covered_3D([x_city, y_city, z_city], satellites_spherical_coordinates, scope) and not has_enough_intensity([x_city, y_city, z_city], satellites_spherical_coordinates, fm.minimum_intensity(height, earth_radius, fm.I)[0], scope)): print("Orange!")
        plotter.add_mesh(pv.Sphere(radius=earth_radius/15, center=(x_city, y_city, z_city)), color=color, point_size=20)

    if kmeans:
        for i, (x_city, y_city, z_city) in enumerate(original_cities_coordinates):
            is_covered = is_covered_3D([x_city, y_city, z_city], satellites_coordinates, scope)
            color = 'pink' if is_covered else 'orange'
            plotter.add_mesh(pv.Sphere(radius=1, center=(x_city, y_city, z_city)), color=color, point_size=20)
            plotter.add_text(str(original_cities_weights[i]), position=(x_city, y_city, z_city), font_size=5)
    
    # Position de la caméra
    plotter.camera_position = [(-250, -250, -250), (0, 0, -1), (0, 1, 0)]

    # Légendage
    plotter.add_text(f"{cities_coordinates.shape[0]} villes, {satellites_coordinates.shape[0]} satellites", position="upper_right", font_size=10, color="white")
    plotter.add_text(f"{np.round(satisfied_proportion*100, decimals=2)} % de la population a une couverture réseau acceptable (I > {np.round(fm.minimum_intensity(height, earth_radius, fm.I)[0], decimals=7)})", position="upper_left", font_size=12, color="white")

    # Show the plot
    plotter.show()
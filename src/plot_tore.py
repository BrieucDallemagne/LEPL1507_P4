import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def has_enough_intensity(city_coords, satellites_coords, min_intensity, scope,coefs):
    total_intensity = 0
    city_coords = np.array(city_coords)
    for satellite_coords_curr, coef in zip(satellites_coords, coefs):        
        city_satellite_distance = math.sqrt(
            (city_coords[0] - satellite_coords_curr[0]) ** 2 + (city_coords[1] - satellite_coords_curr[1]) ** 2 + (city_coords[2] - satellite_coords_curr[2]) ** 2)
        if city_satellite_distance <= scope:
            total_intensity += fm.I(city_satellite_distance, coef=coef)
    return total_intensity >= min_intensity


def plot_torus(cities_coordinates, satellites_coordinates, cities_weights, height, kmeans= False, centroids= np.array(0), centroids_weights=np.array(0),rot=False, planet= "earth"):
    sphere_center = (0, 0, 0)
    earth_radius = 50
    earth_radius2 = 20
    cities_coordinates[:, 0] += np.pi
    cities_coordinates_spherical = np.copy(cities_coordinates)
    cities_coordinates = fm.spherical_to_cartesian_torus(cities_coordinates, sphere_center, earth_radius, earth_radius2)
    if planet == "moon":
        image = "Planet_Images\moon.jpg"
    elif planet == "mars":
        image = "Planet_Images\mars.jpg"
    elif planet == "earth_night":
        image = "Planet_Images\earth_night.jpg"
    elif planet == "Remacle":
        image = "Planet_Images\Remacle.png"
    else:
        image = "Planet_Images\earth.jpg"

    # Créer le plot en 3D
    plotter = pv.Plotter()
    plotter.set_background('black')

     # Rayon du tore
    torus_radius = earth_radius
    torus_cross_section_radius = earth_radius2
    resolution = 100  

    
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0, 2*np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    x = (torus_radius + torus_cross_section_radius * np.cos(phi)) * np.cos(theta)
    y = (torus_radius + torus_cross_section_radius * np.cos(phi)) * np.sin(theta)
    z = torus_cross_section_radius * np.sin(phi)
    tore_mesh = pv.StructuredGrid(x, y, z)

    # Charger la texture
    texture = pv.read_texture("Planet_Images\earth.jpg")

    # Ajouter la texture au maillage du tore
    tore_mesh.texture_map_to_plane(inplace=True)
    tore_mesh.texture = texture
    plotter.add_mesh(tore_mesh,texture=texture)


    # Autre planète (facultatif)
    second_mesh = pv.StructuredGrid(x, y, z)
    second_texture = pv.read_texture("Planet_Images\sun.jpg")
    second_mesh.texture_map_to_plane(inplace=True)
    second_mesh.texture = second_texture

    second_mesh.points = 60*second_mesh.points + np.array([10000, 10000, 10000])
    second_mesh.rotate_x(135)
    second_mesh.rotate_y(45)
    second_mesh.rotate_z(90)
    plotter.add_mesh(second_mesh, texture=second_texture)

    # Rayon de la sphère
    satellite_radius = 50 + height
    scope = fm.find_x(height, earth_radius)*10

    # Dessiner les satellites sur un tore
    satellites_cart_coordinates = fm.spherical_to_cartesian_torus(satellites_coordinates, sphere_center, torus_radius, torus_cross_section_radius+height)
    x_sat = satellites_cart_coordinates[:, 0]
    y_sat = satellites_cart_coordinates[:, 1]
    z_sat = satellites_cart_coordinates[:, 2]

    intensity_matrix = fm.I_tore(cities_coordinates_spherical, cities_coordinates, x_sat, y_sat, z_sat)
    for x_s, y_s, z_s in zip(x_sat, y_sat, z_sat):
        sphere = pv.Sphere(radius=scope/50, center=(x_s, y_s, z_s))
        plotter.add_mesh(sphere, color='firebrick', point_size=10, opacity=0.3)
    satellites_cart_coordinates = np.c_[x_sat, y_sat, z_sat]
    satisfied_proportion = 0
    alpha_coefs = fm.coef_tore(cities_coordinates_spherical, cities_coordinates, x_sat, y_sat, z_sat)
    for i, ((x_city, y_city, z_city), coefs) in enumerate(zip(cities_coordinates, alpha_coefs)):
        is_covered = is_covered_3D([x_city, y_city, z_city], satellites_cart_coordinates, scope)
        if has_enough_intensity([x_city, y_city, z_city], satellites_cart_coordinates, fm.minimum_intensity2(height, fm.I), scope, coefs): satisfied_proportion += cities_weights[i]
        color = 'green' if is_covered_3D([x_city, y_city, z_city], satellites_cart_coordinates, scope) and has_enough_intensity([x_city, y_city, z_city], satellites_cart_coordinates, fm.minimum_intensity2(height, fm.I), scope,coefs) else \
                'orange' if is_covered_3D([x_city, y_city, z_city], satellites_cart_coordinates, scope) and not has_enough_intensity([x_city, y_city, z_city], satellites_cart_coordinates, fm.minimum_intensity2(height, fm.I), scope,coefs) else \
                'red'
        if (is_covered_3D([x_city, y_city, z_city], satellites_cart_coordinates, scope) and not has_enough_intensity([x_city, y_city, z_city], satellites_cart_coordinates, fm.minimum_intensity2(height, fm.I), scope,coefs)): print("Orange!")

        plotter.add_mesh(pv.Sphere(radius=earth_radius/50, center=(x_city, y_city, z_city)), color=color, point_size=20)

    if kmeans:
        for i, (x_og, y_og, z_og) in enumerate(centroids):
            is_covered = is_covered_3D([x_og, y_og, z_og], satellites_cart_coordinates, scope)
            color = 'pink' if is_covered else 'blue'
            plotter.add_mesh(pv.Sphere(radius=earth_radius/15, center=(x_og, y_og, z_og)), color=color, point_size=20)

    
    # Position de la caméra
    plotter.camera_position = [(-250, -250, -250), (0, 0, -1), (0, 1, 0)]

    # Légendage
    plotter.add_text(f"{cities_coordinates.shape[0]} villes, {satellites_coordinates.shape[0]} satellites", position="upper_right", font_size=10, color="white")
    plotter.add_text(f"{np.round(satisfied_proportion*100, decimals=2)} % de la population a une couverture réseau acceptable (I > {np.round(fm.minimum_intensity2(height, fm.I), decimals=7)})", position="upper_left", font_size=12, color="white")

    # Show the plot
    plotter.show()
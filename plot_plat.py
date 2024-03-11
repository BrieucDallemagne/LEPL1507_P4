import numpy as np
import matplotlib.pyplot as plt
import fonction_math as fm
from matplotlib.patches import Circle

def mapping(cities_coordinates, max_grid_size):
    x_min = np.min(cities_coordinates[:, 0])
    x_max = np.max(cities_coordinates[:, 0])
    y_min = np.min(cities_coordinates[:, 1])
    y_max = np.max(cities_coordinates[:, 1])
    cities_coordinates -= np.array([x_min, y_min])
    x_length = x_max - x_min
    y_length = y_max - y_min
    max_length = max(x_length, y_length)
    transformation_ratio = max_grid_size / max_length
    return cities_coordinates * transformation_ratio

def plot_covering_2D(cities_coordinates, cities_weights, satellites_coordinates, grid_size):
    # Extraire les coordonnées x et y des points
    x_coords, y_coords = zip(*cities_coordinates)

    # Villes
    population_proportion = 0
    for x_coord, y_coord, weight in zip(x_coords, y_coords, cities_weights):
        is_covered = fm.is_covered_2D(np.array([x_coord, y_coord]), satellites_coordinates)
        if is_covered:
            population_proportion += weight
        plt.scatter(x_coord, y_coord,
                    color='green' if is_covered else 'red',
                    marker='o',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8)
    
    # Satellites
    for center, radius in zip(satellites_coordinates[:, :2], satellites_coordinates[:, 2]):
        circle = Circle(center, radius, color='red', alpha=0.1)
        plt.gca().add_patch(circle)

    plt.text(0.85, -0.1, f'Population Proportion: {np.round(population_proportion, decimals=4)}',
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)
    # mettre le première arg à 0.1 pour afficher en bas à gauche
        


    plt.scatter(satellites_coordinates[:, 0], satellites_coordinates[:, 1], color='blue', marker='x')

    plt.title(f'Network coverage of {len(cities_coordinates)} cities by {len(satellites_coordinates)} satellites for a radius of {round(radius, 4)}', fontweight='bold')  # Increase the title font size and weight
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.axis('equal')

    plt.grid(color='gray', linestyle='dashed')


    plt.show()








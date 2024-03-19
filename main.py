import spherical_satellites_repartition as ssr
import euclidean_satellites_repartition as esr
import plot_plat as pp
import numpy as np
import matplotlib.pyplot as plt
import random
import fonction_math as fm
import plot_rond as pr
import math
import matplotlib
import tkinter as tk
from tkinter import messagebox

def choisir_mode(kmeans=False, verbose=False):
    mode = mode_var.get()
    num_villes = int(villes_entry.get())

    if mode == "Plat":
        grid_size = random.randint(10, 100)
        n_cities = num_villes
        poids = fm.create_weight(n_cities)
        height = random.randint(1, 10)
        scope = random.randint(height, 20)
        radius = np.sqrt(scope**2 - height**2)

        cities_coordinates = np.random.randint(0, grid_size, size=(n_cities, 2))
        nbr_max_sat = fm.nbr_max_sat(cities_coordinates, grid_size, radius)
        number_of_satellites = nbr_max_sat
        print('nombres de satellites optimal :', nbr_max_sat)
        satellites_coordinates = esr.euclidean_satellites_répartition(number_of_satellites, cities_coordinates, poids, grid_size, scope, height, intensity=1000)
        pp.plot_covering_2D(cities_coordinates, poids, satellites_coordinates, grid_size)
        plt.show()
    elif mode == "Sphérique":
        n_cities = num_villes
        #cities_weights = fm.create_weight(n_cities)
        cities_weights = np.full(n_cities, 1/n_cities)
        radius_earth = 50

        cities_coordinates_latitude = np.random.randint(-90, 90, size=(n_cities))
        cities_coordinates_longitude = np.random.randint(-180, 180, size=(n_cities))
        cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]

        cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
        cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
        cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
        cities_coordinates = np.c_[cities_x, cities_y, cities_z]
        original_cities = cities_coordinates
        original_weights = cities_weights
        number_of_satellites = np.random.randint(1, n_cities)
        satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
        pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights,  10,  original_cities, original_weights, kmeans= False,rot= False)
        plt.show()

def creer_interface():
    global mode_var, villes_entry

    fenetre = tk.Tk()
    fenetre.title("Choix du Mode et du Nombre de Villes")

    fenetre.geometry("400x200") 

    mode_label = tk.Label(fenetre, text="Choisissez le mode:")
    mode_label.pack()

    mode_var = tk.StringVar()
    mode_var.set("Sphérique")  # Par défaut, le mode est plat
    mode_menu = tk.OptionMenu(fenetre, mode_var, "Plat", "Sphérique")
    mode_menu.pack()

    villes_label = tk.Label(fenetre, text="Nombre de villes:")
    villes_label.pack()

    villes_entry = tk.Entry(fenetre)
    villes_entry.pack()

    bouton_valider = tk.Button(fenetre, text="Valider", command=choisir_mode)
    bouton_valider.pack()

    fenetre.mainloop()

def main():
    creer_interface()

if __name__ == "__main__":
    main()



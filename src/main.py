import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import src.spherical_satellites_repartition as ssr
import src.euclidean_satellites_repartition as esr

import plots.plot_plat as pp
import plots.plot_rond as pr
import src.fonction_math as fm

def on_resize(event):
    global background_image, image_label
    # Redimensionner l'image pour correspondre à la taille du canvas
    resized_image = original_image.resize((event.width, event.height), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(resized_image)
    canvas.itemconfig(image_label, image=background_image)

def choisir_mode():
    mode = mode_var.get()
    num_villes = int(villes_entry.get())
    kmeans = kmeans_var.get()
    verbose = verbose_var.get()
    rot = rot_var.get()
    planet_type = planet_var.get()

    if num_villes <= 0:
        messagebox.showerror("Erreur", "Le nombre de villes doit être supérieur à zéro.")
        return

    if mode == "Plat":
        grid_size = np.random.randint(10, 100)
        poids = fm.create_weight(num_villes)
        height = np.random.randint(1, 10)
        scope = np.random.randint(height, 20)
        radius = np.sqrt(scope**2 - height**2)

        cities_coordinates = np.random.randint(0, grid_size, size=(num_villes, 2))
        nbr_max_sat = fm.nbr_max_sat(cities_coordinates, grid_size, radius)
        number_of_satellites = nbr_max_sat
        satellites_coordinates = esr.euclidean_satellites_répartition(number_of_satellites, cities_coordinates, poids, grid_size, scope, height, intensity=1000)
        pp.plot_covering_2D(cities_coordinates, poids, satellites_coordinates, grid_size)
        plt.show()
    elif mode == "Sphérique":
        cities_coordinates_latitude = np.random.randint(-90, 90, size=(num_villes))
        cities_coordinates_longitude = np.random.randint(-180, 180, size=(num_villes))
        cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
        cities_weights = np.full(num_villes, 1/num_villes)

        radius_earth = 50
        cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
        cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
        cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
        cities_coordinates = np.c_[cities_x, cities_y, cities_z]

        original_cities = cities_coordinates
        original_weights = cities_weights

        if kmeans:
            cities_coordinates, cities_weights = fm.k_means_cities(cities_coordinates, num_villes//2, cities_weights)

        number_of_satellites = np.random.randint(1, num_villes)
        satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)
        
        if satellites_coordinates.size == 0:
            messagebox.showerror("Erreur", "Aucun satellite n'a été trouvé.")
            return

        pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights,  10, kmeans=kmeans, centroids=np.array(0), centroids_weights=np.array(0), rot=rot, planet=planet_type)
        plt.show()

def creer_interface():
    global mode_var, villes_entry, kmeans_var, verbose_var, rot_var, canvas, original_image, background_image, image_label, planet_var

    fenetre = tk.Tk()
    fenetre.title("Choix du Mode et du Nombre de Villes")
    
    # Définition de la taille de la fenêtre
    fenetre.geometry("450x350")  # Largeur x Hauteur

    # Charger l'image d'arrière-plan
    original_image = Image.open("Planet_Images/Remacle.png")
    original_image = original_image.resize((450, 300), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(original_image)

    # Créer un canvas pour afficher l'image
    canvas = tk.Canvas(fenetre, width=450, height=300)
    canvas.pack(fill="both", expand=True)
    image_label = canvas.create_image(0, 0, image=background_image, anchor="nw")

    # Liens pour redimensionner l'image en cas de redimensionnement de la fenêtre
    canvas.bind("<Configure>", on_resize)

    # Couleurs
    couleur_texte = "#000000"

    # Titre
    titre_label = tk.Label(canvas, text="Sélection des paramètres", bg="white", fg=couleur_texte)
    titre_label.place(relx=0.5, rely=0.1, anchor="center")

    # Choix du mode
    mode_label = tk.Label(canvas, text="Mode:", bg="white", fg=couleur_texte)
    mode_label.place(relx=0.3, rely=0.3, anchor="center")

    mode_var = tk.StringVar()
    mode_var.set("Sphérique")  # Par défaut, le mode est sphérique
    mode_menu = tk.OptionMenu(canvas, mode_var, "Plat", "Sphérique")
    mode_menu.config(bg="white", fg=couleur_texte)
    mode_menu.place(relx=0.7, rely=0.3, anchor="center")

    # Nombre de villes
    villes_label = tk.Label(canvas, text="Nombre de villes:", bg="white", fg=couleur_texte)
    villes_label.place(relx=0.3, rely=0.4, anchor="center")

    villes_entry = tk.Entry(canvas)
    villes_entry.place(relx=0.7, rely=0.4, anchor="center")

    # KMeans
    kmeans_label = tk.Label(canvas, text="KMeans:", bg="white", fg=couleur_texte)
    kmeans_label.place(relx=0.3, rely=0.5, anchor="center")

    kmeans_var = tk.BooleanVar()
    kmeans_checkbox = tk.Checkbutton(canvas, text="Activer", variable=kmeans_var, bg="white", fg=couleur_texte)
    kmeans_checkbox.place(relx=0.7, rely=0.5, anchor="center")

    # Rotation
    rot_label = tk.Label(canvas, text="Rotation:", bg="white", fg=couleur_texte)
    rot_label.place(relx=0.3, rely=0.6, anchor="center")

    rot_var = tk.BooleanVar()
    rot_checkbox = tk.Checkbutton(canvas, text="Activer", variable=rot_var, bg="white", fg=couleur_texte)
    rot_checkbox.place(relx=0.7, rely=0.6, anchor="center")

    # Verbose
    verbose_label = tk.Label(canvas, text="Verbose:", bg="white", fg=couleur_texte)
    verbose_label.place(relx=0.3, rely=0.7, anchor="center")

    verbose_var = tk.BooleanVar()
    verbose_checkbox = tk.Checkbutton(canvas, text="Activer", variable=verbose_var, bg="white", fg=couleur_texte)
    verbose_checkbox.place(relx=0.7, rely=0.7, anchor="center")

    # Choix du type de planète
    planet_label = tk.Label(canvas, text="Type de planète:", bg="white", fg=couleur_texte)
    planet_label.place(relx=0.3, rely=0.8, anchor="center")

    planet_var = tk.StringVar()
    planet_var.set("earth")  # Par défaut, Remacle est sélectionné
    planet_menu = tk.OptionMenu(canvas, planet_var, "earth", "earth_night", "moon", "mars", "Remacle")
    planet_menu.config(bg="white", fg=couleur_texte)
    planet_menu.place(relx=0.7, rely=0.8, anchor="center")

    # Bouton de validation
    bouton_valider = tk.Button(canvas, text="Valider", command=choisir_mode, bg="white", fg=couleur_texte)
    bouton_valider.place(relx=0.5, rely=0.9, anchor="center")

    fenetre.mainloop()

def main():
    creer_interface()

if __name__ == "__main__":
    main()

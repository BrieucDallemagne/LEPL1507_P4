import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import spherical_satellites_repartition as ssr
import euclidean_satellites_repartition as esr
import plot_plat as pp
import plot_rond as pr
import fonction_math as fm


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
class SatelliteApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Remacle & Associates")
        self.geometry("400x400")
        self.resizable(True, True)
        self.set_background()
        self.create_widgets()

    def set_background(self):
        # Changer d'iamge ici
        image = Image.open("bg.png")
        image = image.resize((800, 700), Image.ANTIALIAS)
        background_image = ImageTk.PhotoImage(image)
        background_label = tk.Label(self, image=background_image)
        background_label.image = background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
    def create_widgets(self):

        # Mode
        self.mode_var = tk.StringVar(self)  # Variable de contrôle
        self.mode_var.set("Plat")  # Option par défaut
        self.mode_label = ctk.CTkLabel(self, text="Mode")
        self.mode_label.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        # Création du menu déroulant avec les options
        self.mode_menu = ctk.CTkOptionMenu(master=self, values=["Plat", "Sphérique"], variable=self.mode_var)
        self.mode_menu.grid(row=1, column=1, padx=20, pady=20, columnspan=2, sticky="ew")

        # Number of cities
        self.number_cities_label = ctk.CTkLabel(self, text="Number of Cities")
        self.number_cities_label.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        self.number_cities_entry = ctk.CTkEntry(self,
                                     placeholder_text="18")
        self.number_cities_entry.grid(row=2, column=1,
                           columnspan=3, padx=20,
                           pady=20, sticky="ew")

        # K-Means
        self.kmeans_label = ctk.CTkLabel(self, text="Kmeans")
        self.kmeans_label.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        self.kmeans_checkbox_var = ctk.BooleanVar()
        self.kmeans_checkbox = ctk.CTkCheckBox(self, text="Kmeans", variable=self.kmeans_checkbox_var)
        self.kmeans_checkbox.grid(row=3, column=1, padx=20, pady=20, sticky="ew")

        # Rotation
        self.rot_label = ctk.CTkLabel(self, text="Rotation")
        self.rot_label.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        self.rot_checkbox_var = ctk.BooleanVar()
        self.rot_checkbox = ctk.CTkCheckBox(self, text="Rotation", variable=self.rot_checkbox_var)
        self.rot_checkbox.grid(row=4, column=1, padx=20, pady=20, sticky="ew")

        # Verbosity
        self.verbose_label = ctk.CTkLabel(self, text="Verbose")
        self.verbose_label.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

        self.verbose_checkbox_var = ctk.BooleanVar()
        self.verbose_checkbox = ctk.CTkCheckBox(self, text="Verbose", variable=self.verbose_checkbox_var)
        self.verbose_checkbox.grid(row=5, column=1, padx=20, pady=20, sticky="ew")

        # Planet

        #self.planet_var = ctk.StringVar()
        #self.planet_var.set("earth")  # Par défaut, Remacle est sélectionné
        #self.planet_menu = ctk.CTkOptionMenu(self, self.planet_var, "earth", "earth_night", "moon", "mars", "Remacle")
        #self.planet_option_menu.grid(row=6, column=1, padx=20, pady=20, columnspan=2, sticky="ew")

        # Generate Button
        self.generate_results_button = ctk.CTkButton(self, text="Generate Results", command=self.choisir_mode)
        self.generate_results_button.grid(row=7, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

        self.reset_results_button = ctk.CTkButton(self, text="Generate Results", command=self.choisir_mode)
        self.reset_results_button.grid(row=7, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

    def choisir_mode(self):
        mode = self.mode_var.get()
        num_villes = int(self.number_cities_entry.get())
        kmeans = self.kmeans_checkbox_var.get()
        verbose = self.verbose_checkbox_var.get()
        rot = self.rot_checkbox_var.get()
        print(mode)
        #planet_type = self.planet_option_menu.get()

        if num_villes <= 0:
            messagebox.showerror("Erreur", "Le nombre de villes doit être supérieur à zéro.")
            return
        print("hello")
        if mode == "Plat":
            grid_size = np.random.randint(10, 100)
            poids = fm.create_weight(num_villes)
            height = np.random.randint(1, 10)
            scope = np.random.randint(height, 20)
            radius = np.sqrt(scope ** 2 - height ** 2)

            cities_coordinates = np.random.randint(0, grid_size, size=(num_villes, 2))
            nbr_max_sat = fm.nbr_max_sat(cities_coordinates, grid_size, radius)
            number_of_satellites = nbr_max_sat
            satellites_coordinates = esr.euclidean_satellites_répartition(number_of_satellites, cities_coordinates,
                                                                          poids, grid_size, scope, height,
                                                                          intensity=1000)
            pp.plot_covering_2D(cities_coordinates, poids, satellites_coordinates, grid_size)
            plt.show()
        elif mode == "Sphérique":
            print("hello")
            cities_coordinates_latitude = np.random.randint(-90, 90, size=(num_villes))
            cities_coordinates_longitude = np.random.randint(-180, 180, size=(num_villes))
            cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]
            cities_weights = np.full(num_villes, 1 / num_villes)

            radius_earth = 50
            cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in
                        cities_coordinates]
            cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in
                        cities_coordinates]
            cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
            cities_coordinates = np.c_[cities_x, cities_y, cities_z]

            original_cities = cities_coordinates
            original_weights = cities_weights

            if kmeans:
                cities_coordinates, cities_weights = fm.k_means_cities(cities_coordinates, num_villes // 2,
                                                                       cities_weights)

            number_of_satellites = np.random.randint(1, num_villes)
            satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10,
                                                                          verbose=verbose)

            if satellites_coordinates.size == 0:
                messagebox.showerror("Erreur", "Aucun satellite n'a t trouv.")
                return

            pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=kmeans, rot=rot)
            plt.show()

if __name__ == "__main__":
    app = SatelliteApp()
    app.mainloop()

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import src.spherical_satellites_repartition as ssr
import src.euclidean_satellites_repartition as esr
import pandas as pd
import tests.test_rond as tr
import pandas as pd
import src.resolve_csv as rc

import plots.plot_plat as pp
import plots.plot_rond as pr
import src.fonction_math as fm


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


class SatelliteApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Remacle & Associates")
        self.geometry("600x600")
        self.resizable(True, True)
        self.background_label = tk.Label(self)
        self.set_background()
        self.create_widgets()
        self.bind("<Configure>", self.resize_background)

    def set_background(self):
        # Changer d'iamge ici
        self.image = Image.open("Planet_Images/beautiful_bg.jpg")
        self.update_background()

    def update_background(self):
        resized_image = self.image.resize((self.winfo_width(), self.winfo_height()), Image.LANCZOS)
        background_image = ImageTk.PhotoImage(resized_image)
        self.background_label.config(image=background_image)
        self.background_label.image = background_image
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

    def resize_background(self, event):
        self.update_background()

    def create_widgets(self):

        # Mode
        self.mode_var = tk.StringVar(self)  # Variable de contrôle
        self.mode_var.set("Sphérique")  # Option par défaut
        self.mode_label = ctk.CTkLabel(self, text="Mode")
        self.mode_label.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        # Création du menu déroulant avec les options
        self.mode_menu = ctk.CTkOptionMenu(master=self, values=["Plat", "Sphérique","réel","csv_plat","csv_rond"], variable=self.mode_var)
        self.mode_menu.grid(row=1, column=1, padx=20, pady=20, columnspan=2, sticky="ew")

        # Number of cities
        self.number_cities_label = ctk.CTkLabel(self, text="Number of Cities")
        self.number_cities_label.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        self.number_cities_entry = ctk.CTkEntry(self,
                                     placeholder_text="20")
        self.number_cities_entry.grid(row=2, column=1,
                           columnspan=3, padx=20,
                           pady=20, sticky="ew")
        
        # Number of cities
        self.csvname_label = ctk.CTkLabel(self, text="csv name")
        self.csvname_label.grid(row=6, column=0, padx=20, pady=20, sticky="ew")
        self.csvname = ctk.CTkEntry(self,
                                     placeholder_text="csv name (+ .csv)")
        self.csvname.grid(row=6, column=1,
                           columnspan=3, padx=20,
                           pady=20, sticky="ew")

        # K-Means
        self.kmeans_label = ctk.CTkLabel(self, text="Kmeans")
        self.kmeans_label.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        self.kmeans_checkbox_var = ctk.BooleanVar()
        self.kmeans_checkbox = ctk.CTkCheckBox(self, text="Kmeans", variable=self.kmeans_checkbox_var)
        self.kmeans_checkbox.grid(row=3, column=1, padx=20, pady=20, sticky="ew")
        # Verbosity
        self.verbose_label = ctk.CTkLabel(self, text="Verbose")
        self.verbose_label.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

        self.verbose_checkbox_var = ctk.BooleanVar()
        self.verbose_checkbox = ctk.CTkCheckBox(self, text="Verbose", variable=self.verbose_checkbox_var)
        self.verbose_checkbox.grid(row=5, column=1, padx=20, pady=20, sticky="ew")

        self.real_conditions_button = ctk.CTkButton(self, text="Conditions réelles", command=self.set_real_conditions)
        self.real_conditions_button.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        # Planet

        #self.planet_var = ctk.StringVar()
        #self.planet_var.set("earth")  # Par défaut, Remacle est sélectionné
        #self.planet_menu = ctk.CTkOptionMenu(self, self.planet_var, "earth", "earth_night", "moon", "mars", "Remacle")
        #self.planet_option_menu.grid(row=6, column=1, padx=20, pady=20, columnspan=2, sticky="ew")

        # Generate Button
        self.generate_results_button = ctk.CTkButton(self, text="Generate Results", command=self.choisir_mode)
        self.generate_results_button.grid(row=8, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

        self.reset_results_button = ctk.CTkButton(self, text="Generate Results", command=self.choisir_mode)
        self.reset_results_button.grid(row=8, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

    def set_real_conditions(self):
        self.mode_var.set("réel")
        if self.number_cities_entry.get() == '':
            self.number_cities_entry.insert(0, "18")
        self.verbose_checkbox_var.set(False)
        self.generate_results_button.invoke()

    def choisir_mode(self):
        mode = self.mode_var.get()
        csvname = self.csvname.get()
        if self.number_cities_entry.get() == '':
            self.number_cities_entry.insert(0, "18")
        num_villes = int(self.number_cities_entry.get())
        kmeans = self.kmeans_checkbox_var.get()
        verbose = self.verbose_checkbox_var.get()
        #planet_type = self.planet_option_menu.get()

        if num_villes <= 0:
            messagebox.showerror("Erreur", "Le nombre de villes doit être supérieur à zéro.")
            return

        if mode == "Plat":
            grid_size = np.random.randint(10, 100)
            poids = fm.create_weight(num_villes)
            height = np.random.randint(1, 10)
            cities_weights = np.full(num_villes, 1 / num_villes)
            scope = np.random.randint(height, 20)
            radius = np.sqrt(scope ** 2 - height ** 2)
            cities_coordinates = np.random.randint(0, grid_size, size=(num_villes, 2))
            nbr_max_sat = fm.nbr_max_sat(cities_coordinates, grid_size, radius)
            number_of_satellites = nbr_max_sat
            if kmeans:
                centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, num_villes // 2,
                                                                         cities_weights)
                satellites_coordinates = esr.euclidean_satellites_repartition(number_of_satellites, centroids_coordinates,
                                                                          centroids_weights, grid_size, scope, height,
                                                                          intensity=1000)
            else:
                satellites_coordinates = esr.euclidean_satellites_repartition(number_of_satellites, cities_coordinates,
                                                                          poids, grid_size, scope, height,
                                                                          intensity=1000)
            pp.plot_covering_2D(cities_coordinates, poids, satellites_coordinates)
            plt.show()
        elif mode == "Sphérique":
            cities_coordinates_latitude = np.random.randint(-90, 90, size=(num_villes))
            cities_coordinates_longitude = np.random.randint(-180, 180, size=(num_villes))
            cities_coordinates = np.c_[cities_coordinates_longitude, cities_coordinates_latitude]
            cities_weights = np.full(num_villes, 1 / num_villes)

            if kmeans:
                centroids_coordinates, centroids_weights = fm.k_means_cities(cities_coordinates, num_villes//2, cities_weights, spherical=True)
                satellites_coordinates = ssr.spherical_satellites_repartition(fm.cartesian_to_spherical(centroids_coordinates), centroids_weights, 10, verbose=verbose)
            else:
                satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=verbose)

            if satellites_coordinates.size == 0:
                messagebox.showerror("Erreur", "Aucun satellite n'a été trouvé")
                return
            
            if kmeans:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=True, centroids=centroids_coordinates)
            else:
                pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False)
            plt.show()

        elif mode == "réel":
            tr.test_solve_3D_random(n_cities=num_villes,n_tests=1, k_means=False, 
                                    real_cities = True, verbose = verbose, planet=False)
            
        elif mode == "csv_plat":
            rc.resolve_carre(num_villes, csvname)

        elif mode == "csv_rond":
            rc.resolve_rond(num_villes, csvname)

            

if __name__ == "__main__":
    app = SatelliteApp()
    app.mainloop()

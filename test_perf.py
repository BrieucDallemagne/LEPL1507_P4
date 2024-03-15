import test_plat as fm
import time
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import spherical_satellites_repartition as ssr
import numpy as np
from time import perf_counter
import seaborn as sns
sns.set_theme()

scope_size_list = [100, 200, 300, 500, 1000,3000, 5000]
cities_size_list = [2, 5, 10, 20, 50, 100]
grd_size_list = [10, 50, 100, 300, 500, 750, 1000]
def test_performance(list, x, y, save, test_type, xscale = 'log'):
    perf = []
    for i in list:

        if test_type == "n":
            start = time.perf_counter()
            fm.test_solve_2D_random(n=i)
            end = time.perf_counter()
        elif test_type == "scope":
            start = time.perf_counter()
            fm.test_solve_2D_random(scope=i)
            end = time.perf_counter()
        elif test_type == "grid_size":
            start = time.perf_counter()
            fm.test_solve_2D_random(grid_size=i)
            end = time.perf_counter()
        elapsed_time = end - start
        perf.append(elapsed_time)
        print("Temps consacré pour", i, ":", elapsed_time)
    plt.figure()
    plt.plot(list, perf, 'o-', label = "Temps d'exécution", color='navy')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, ls = '--', lw = 0.5)
    #save la figure
    plt.savefig(save+'.png')
    plt.show()

def plot_complexity_solver():
    N = [i for i in range(40, 1000, 40)]
    perf = []
    for n in N:
        print(n)
        t = [0, 0, 0]
        for i in range(3):
            cities_weights = np.full(n, 1/n)
            radius_earth = 50

            cities_coordinates_latitude = np.linspace(-90, 90, n)
            cities_coordinates_longitude = np.linspace(-180, 180, n)
            cities_coordinates = np.c_[cities_coordinates_latitude, cities_coordinates_longitude]

            cities_x = [radius_earth * np.cos(np.radians(coord[1])) * np.cos(np.radians(coord[0])) for coord in cities_coordinates]
            cities_y = [radius_earth * np.cos(np.radians(coord[1])) * np.sin(np.radians(coord[0])) for coord in cities_coordinates]
            cities_z = [radius_earth * np.sin(np.radians(coord[1])) for coord in cities_coordinates]
            cities_coordinates = np.c_[cities_x, cities_y, cities_z]
            start = time.perf_counter()
            ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=False)
            end = time.perf_counter()
            t[i] = end - start
        perf.append(np.min(t))

    plt.figure()
    plt.plot(N, perf, 'o-', label = "Temps d'exécution", color='navy', markersize=2)
    plt.grid(True, ls = '--', lw = 0.5)
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps (s)")
    plt.show()
    plt.savefig('performance_solver.pdf')




#test_performance(cities_size_list, "Nombre de villes", "Temps (s)", "performance_villes", "n")
#test_performance(scope_size_list,  "Portée", "Temps (s)", "performance_portee", "scope")
#test_performance(grd_size_list, "Taille de la grille", "Temps (s)", "performance_grille", "grid_size")
plot_complexity_solver()

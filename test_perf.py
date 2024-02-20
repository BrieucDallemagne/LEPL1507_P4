import test_plat as fm
import time
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
mplstyle.use(['dark_background', 'ggplot', 'fast'])
"un max de points"
scope_size_list = [100, 200, 300, 500, 1000,3000, 5000]
cities_size_list = [2, 5, 10, 20, 50, 100]
grd_size_list = [10, 50, 100, 300, 500, 750, 1000]
def test_performance(list, x, y, save, test_type, xscale = 'log'):
    perf = []
    for i in list:

        if test_type == "n_cities":
            start = time.perf_counter()
            fm.test_solve_2D_random(n_cities=i)
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


test_performance(cities_size_list, "Nombre de villes", "Temps (s)", "performance_villes", "n_cities")
#test_performance(scope_size_list,  "Portée", "Temps (s)", "performance_portee", "scope")
#test_performance(grd_size_list, "Taille de la grille", "Temps (s)", "performance_grille", "grid_size")

import numpy as np
import pyvista as pv

def compute_torus_normals(theta, phi):
    """
    Compute normals to the surface of a torus at given theta and phi values.

    Parameters:
    - theta: Angle theta in radians (array)
    - phi: Angle phi in radians (array)

    Returns:
    - normals: Normals to the surface of the torus at given theta and phi values (array)
    """
    normal_x = np.cos(theta) * np.cos(phi)
    normal_y = np.sin(theta) * np.cos(phi)
    normal_z = np.sin(phi)
    return np.array([normal_x, normal_y, normal_z]).T

# Paramètres du tore
major_radius = 3
minor_radius = .5
num_points = 10

# Générer des angles theta et phi aléatoires
np.random.seed(0)  # Pour la reproductibilité
theta = np.random.uniform(0, 2*np.pi, num_points)
phi = np.random.uniform(0, 2*np.pi, num_points)

# Créer un maillage PyVista pour le tore
resolution = 50
theta_grid, phi_grid = np.meshgrid(np.linspace(0, 2*np.pi, resolution), np.linspace(0, 2*np.pi, resolution))
x = (major_radius + minor_radius * np.cos(phi_grid)) * np.cos(theta_grid)
y = (major_radius + minor_radius * np.cos(phi_grid)) * np.sin(theta_grid)
z = minor_radius * np.sin(phi_grid)
mesh = pv.StructuredGrid(x, y, z)

# Calculer les normales à la surface du tore pour les angles donnés
normals = compute_torus_normals(theta, phi)

# Créer un plotter PyVista
p = pv.Plotter()

# Ajouter le maillage avec les normales aux points sélectionnés
p.add_mesh(mesh, color="blue", show_edges=True)

# Ajouter les flèches représentant les normales aux points sélectionnés
p.add_arrows(np.column_stack([(major_radius + minor_radius * np.cos(phi)) * np.cos(theta),
                               (major_radius + minor_radius * np.cos(phi)) * np.sin(theta),
                               minor_radius * np.sin(phi)]),
             normals, mag=0.2, color="red")

# Afficher la visualisation
p.show()

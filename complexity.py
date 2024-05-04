import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import random

import src.spherical_satellites_repartition as ssr
import plots.plot_rond as pr
import tests.test_rond as tr

min_num_villes = 10
max_num_villes = 300
step = 10
num_villes_values = np.arange(min_num_villes, max_num_villes + 1, step)

mean_execution_times = []

for num_villes in num_villes_values:
	print(f"Nombre de villes: {num_villes}")
	execution_times = []

	for i in range(5):
		print(f"Test {i+1}/{5}")
		"""cities_coordinates_latitude = np.random.randint(-90, 90, size=(num_villes))
		cities_coordinates_longitude = np.random.randint(-180, 180, size=(num_villes))
		cities_coordinates = np.c_[cities_coordinates_longitude, cities_coordinates_latitude]
		cities_weights = np.full(num_villes, 1 / num_villes)"""
		file = open('worldcities.csv', 'r', encoding='utf-8')
		csv_reader = csv.DictReader(file)

		longitudes = []
		latitudes = []
		populations = []
		cities_names = []
		countries_names = []

		num_rows = sum(1 for row in csv_reader)
		file.seek(0)
		random_indices = random.sample(range(num_rows), num_villes)

		for i, row in enumerate(csv_reader):
			if i in random_indices:
				longitudes.append(float(row['lng']))
				latitudes.append(float(row['lat']))
				populations.append(row['population'])
				cities_names.append(row['city'])
				countries_names.append(row['country'])

		cities_coordinates_sph = np.array([longitudes, latitudes]).T
		cities_coordinates_sph[:, 1] = (90 - cities_coordinates_sph[:, 1]) 
		cities_coordinates = np.radians(cities_coordinates_sph)
		cities_weights = np.full(cities_coordinates_sph.shape[0], 1/cities_coordinates_sph.shape[0])

		start_time = time.time()
		try:
			satellites_coordinates = ssr.spherical_satellites_repartition(cities_coordinates, cities_weights, 10, verbose=False)
		except:
			continue
		end_time = time.time()
		execution_time = end_time - start_time
		execution_times.append(execution_time)
		print(f"Temps d'exécution: {execution_time}")
		#pr.plot_3D(cities_coordinates, satellites_coordinates, cities_weights, 10, kmeans=False)

	mean_execution_time = np.mean(execution_times)

	mean_execution_times.append(mean_execution_time)

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.loglog(num_villes_values, mean_execution_times, marker='o', linestyle='-')
plt.xlabel('Nombre de villes')
plt.ylabel('Temps d\'exécution moyen (s)')
plt.title('Temps d\'exécution moyen en fonction du nombre de villes')
plt.grid(True)
plt.show()
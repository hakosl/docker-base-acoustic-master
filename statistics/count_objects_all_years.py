import numpy as np

from data.echogram import get_echograms
import utils.plotting
plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt

freqs = [18, 38, 120, 200]
echs = get_echograms(frequencies=freqs, minimum_shape=256)

years = sorted(list(set([ech.year for ech in echs])))
echs_year = {year: [ech for ech in echs if ech.year == year] for year in years}
n_objects = np.zeros((len(years), 3))

for i, year in enumerate(years):
    n_objects_year = {1: 0, 27: 0, -1: 0}
    for ech in echs_year[year]:
        for obj in ech.objects:
            idx = obj["fish_type_index"]
            if idx not in [1, 27]:
                idx = -1
            n_objects_year[idx] += 1
    print(year, n_objects_year)
    n_objects[i, 0] = n_objects_year[27]
    n_objects[i, 1] = n_objects_year[1]
    n_objects[i, 2] = n_objects_year[-1]

n_objects = np.vstack((n_objects, np.sum(n_objects, axis=0)))
n_objects = n_objects.astype(np.int)
print(n_objects)
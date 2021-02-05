import numpy as np

from data.echogram import get_echograms

# NR specific (?) setup for matplotlib
import utils.plotting
plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt


def plot_species(years):

    if type(years) == int:
        years = [years]
    freqs = [18, 38, 120, 200]
    freq_relative = 38
    freq_threshold = 200
    idx_freq_relative = freqs.index(freq_relative)
    idx_freq_threshold = freqs.index(freq_threshold)

    #label_types = [1, 12, 27, 5027, 6007, 6008, 6009, 6010, 9999]
    label_types = [1, 27]
    echograms = get_echograms(years=years)
    years = sorted(list(set([ech.year for ech in echograms])))

    n_pixels_threshold = 0 # Only use shoals with at least this many pixels (after thresholding)
    n_points_plot = 500 # Limit the number of plotting points per species
    pixel_value_threshold = 1e-6 # Remove pixels from label if pixel value below this value (for freq == freq_threshold)
    log_eps = 1e-25 # Avoid log(0)

    for year in years:

        print("\n\n\n", "Year: " + str(year), "\n")
        echograms_year = [ech for ech in echograms if ech.year == year]
        stats = {key: [] for key in label_types}
        n_total = len(echograms_year)

        for i, ech in enumerate(echograms_year):

            if i % 100 == 0:
                print(n_total, i)

            data = ech.data_numpy(frequencies=freqs)
            data[np.invert(np.isfinite(data))] = 0

            if ech.n_objects == 0:
                continue

            for obj in ech.objects:

                fish_type = obj['fish_type_index']

                if obj["n_pixels"] < n_pixels_threshold or fish_type not in label_types:
                    continue

                ind = obj['indexes']
                fish_data = data[ind[:, 0], ind[:, 1]].copy()
                fish_data = fish_data[fish_data[:, idx_freq_threshold] > pixel_value_threshold]

                if fish_data.shape[0] <= n_pixels_threshold:
                    continue

                mean = np.mean(fish_data, axis=0).reshape(-1)
                stats[fish_type].append(mean.tolist())

        for key in list(stats):  # 'list(stats)' (instead of just 'stats') to avoid iterator error when deleting keys

            # Delete empty keys from stats (i.e. fish types that does not occur in the data
            if len(stats[key]) == 0:
                del stats[key]
                continue

            stats[key] = np.array(stats[key])
            np.random.seed(5)
            np.random.shuffle(stats[key])
            stats[key] = np.log10(stats[key][:n_points_plot, :] + log_eps) # Limit the number of points to n_points_plot
            stats[key] = stats[key] - stats[key][:, idx_freq_relative].reshape(-1, 1)  # Pixel values relative to 38 kHz
            print(key, "\tshape\t", stats[key].shape)

        # Plot
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(year, fontsize=20)
        for i_y in range(len(freqs)):
            for i_x in range(len(freqs)):
                if i_y == i_x:
                    continue
                ax = fig.add_subplot(len(freqs), len(freqs), len(freqs) * i_y + i_x + 1)
                for key in stats:
                    z = stats[key][:, (i_x, i_y)]
                    if key == 27:
                        ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.3, c="black") # Black: Sandeel
                    else:
                        ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.3)
                ax.set_xlabel(str(freqs[i_x]) + " kHz", fontsize=12)
                ax.set_ylabel(str(freqs[i_y]) + " kHz", fontsize=12)
        plt.show()


if __name__ == "__main__":
    #plot_species(years=[2015, 2017])
    plot_species(years=[2018])


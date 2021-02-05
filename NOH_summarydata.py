import numpy as np
from data.echogram import get_echograms


def plot_species(years):
    if type(years) == int:
        years = [years]
    freqs = [18, 38, 200, 333]
    #freqs = [38, 120, 200]
    label_types = [1, 12, 27, 5027, 6007, 6008, 6009, 6010, 9999]

    echograms = get_echograms(years=years,frequencies=freqs)
    print(years)
    print(echograms[1].frequencies)
    # count objects

    a=[]
    for ec in echograms:
        #N=N+ec.n_objects
        #print(ec.objects[1]['fish_type_index'])
        for eco in ec.objects:
            try:
                a.extend([eco['fish_type_index']])
            except:
                print('No code')
            
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts

if __name__ == "__main__":
    
    allyears=[2018, 2018]
    for year in allyears:
        unique, counts = plot_species(years=year)
    print(unique)
    print(counts)
    print(np.sum(counts))


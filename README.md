# NR/IMR Environment variables


## IMR Scratch disk
fld_scratch = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/'

## IMR Data storage
fld_storage = '/data/deep/data/echosounder/akustikk_all/'

# Preprocessing data (preprocessing/)

## Generating mat files from IMR folder structure [matlab]

preproces_main.m (previously named generate_mat_files.m)

The preproces_main.m is a matlabscript that reads the .raw and .work files from the 
IMR data storage. The script are run on the IMR unix server and generates mat files stored in the 
X folder.

### Convert a single raw file to mat file [matlab]

generate_mat.m (previously generate_mat_files.m)

The generate_mat.m function reads a snap and a raw file and generate a mat file that contain the range vector, 
time vector, interpretation mask, and sv data.

## Generate memmap-files from mat files [python]

generate_memmap_files.py (previously generate_memmap_files.py)

The generate_memmap_files.py function reads the mat files and generates the memmap files.

# Models (models/)
A folder that contains one file for each model.

# Read training set, define and train model (toplevel folder)

## echogram.py
contains a class that represent each echogram 
This should also contain a function that loads all echogram for a given year/years with given frequencies.

## batch.py
contains functions to load batches for training and validation

## heatmap.py
takes a model and a survey object and predicts for the full echogram



import itertools
import netCDF4 as ncf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import math
import csv
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback
import os
import random
import glob

def get_index(grid,val):
    try:
        return np.where(grid==val)[0][0]
    except (TypeError,IndexError):
            if(len(val)>1):
                index_list=[]
                for i in val:
                    print(type(i))
                    index_list.append(np.where(grid==i)[0][0])
                return index_list
loc_y = [165,121,171]
loc_x = [67,58,39]
grid_size_x = dx = 15
grid_size_y = dy = 15

grids_x = [np.arange(x-dx, x+dx, 1) for x in loc_x]
grids_y = [np.arange(y-dy, y+dy, 1) for y in loc_y]

all_grids_x = np.concatenate(grids_x)
all_grids_y = np.concatenate(grids_y)

all_points = list(itertools.product(all_grids_x, all_grids_y))

extract_x = [point[0] for point in all_points]
extract_y = [point[1] for point in all_points]

monitors_file = 'C:/air_quality_project/weather_data/monitor_list_NC.csv'
with open(monitors_file) as csvfile:
    reader = csv.DictReader(csvfile)
    monitor_points = [(int(float(row['Col'])), int(float(row['Row'] ))) for row in reader if row['Col']] 

monitors_x, monitors_y = zip(*monitor_points)
all_grids_x=np.concatenate([all_grids_x,monitors_x])
all_grids_y=np.concatenate([all_grids_y,monitors_y])


met_dir='C:/air_quality_project/weather_data/met_data/'
emis_dir='C:/air_quality_project/weather_data/emis_data/'
aq_dir='C:/air_quality_project/weather_data/aq_conc/'
short_dir='C:/air_quality_project/weather_data/short_data/'

def get_file_list(data_dir):
    return [os.path.join(data_dir, name) for name in os.listdir(data_dir)]

emis_files = get_file_list(emis_dir)
met_files = get_file_list(met_dir)
aq_files = get_file_list(aq_dir)



emis=['NO', 'NO2']
met=['PBL', 'Q2', 'TEMP2', 'WSPD10', 'WDIR10']
no_of_files=len(met_files)

for i in range(no_of_files):
    emis_file=ncf.Dataset(emis_files[i])
    met_file=ncf.Dataset(met_files[i])
    aq_file=ncf.Dataset(aq_files[i])
    outfile= short_dir+str(i+1)+'.npz'
    time=np.arange(0,25,1,dtype=np.int64)
    time_aq=np.arange(0,24,1,dtype=np.int64)
    
    def create_npz(file,var):
        data_ext=file[var][:,0,all_grids_x,all_grids_y]
        return data_ext
    np.savez(outfile, NO=create_npz(emis_file,'NO'),NO2=create_npz(emis_file,'NO2'),PBL=create_npz(met_file,'PBL')
            ,Q2=create_npz(met_file,'Q2'),TEMP2=create_npz(met_file,'TEMP2'),WSPD10=create_npz(met_file,'WSPD10')
            ,WDIR10=create_npz(met_file,'WDIR10'),O3=create_npz(aq_file,'O3'))
short_files = get_file_list(short_dir)

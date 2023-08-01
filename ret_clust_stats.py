#!/usr/bin/python3

import gsd.hoomd
import hoomd 
import numpy as np
import os
import sys

from scipy.cluster.hierarchy import *
from scipy.spatial.distance import *

l = len(sys.argv)

if l<5:
    print("\n Correct syntax is [ret_clust_stats.py] [delta t] [save_freq] [disk_radius] [path to input directory]\n")
    exit(0)

else :
    pass

dt=float(sys.argv[1])       # timestep width used in simulation 
save_freq=int(sys.argv[2])  # frequency at which simulation data is stored
rad=float(sys.argv[3])      # radius of disks in the simulation
inp_dir=sys.argv[4]       # path to input file, containing the raw, wrapped positions


### particle dimensions
dia=2.*rad
contact_dist=1.05*dia

full_path_inp_dir=os.getcwd() + "/" + inp_dir

directory = os.fsencode(full_path_inp_dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".gsd"):
        basename, file_extension = os.path.splitext(filename)
        larg_clust_name = full_path_inp_dir + "/largest_clust_" + basename + ".dat"
        num_clust_name = full_path_inp_dir + "/num_clust_" + basename + ".dat"
        print(filename, " is the file I am reading")

        full_filename=full_path_inp_dir + "/" + filename

        f=gsd.hoomd.open(full_filename,mode='rb')
        num_frames = f.file.nframes
        box = f[0].configuration.box[:2]
        L = box[0]

        largest_clust_dat=np.zeros((num_frames,2))
        num_clust_dat=np.zeros((num_frames,2))

        ct=0
        for i in range(num_frames):
            ct=ct+1
            f_i=f[i]
            print("reading frame number ", ct)
            pos = f[i].particles.position
            r = pos - pos[:,np.newaxis]
            r[r< -L/2] += L
            r[r > L/2] -= L
            d = np.sqrt(np.sum(r**2, axis=2))
            Y = squareform(d)
            Z = linkage(Y)
            T = fcluster(Z, t=contact_dist, criterion='distance')
            unique, counts = np.unique(T, return_counts=True)
            num_clust_dat[i,0] = ct*dt*save_freq
            num_clust_dat[i,1] = len(unique)
            largest_clust_dat[i,0] = ct*dt*save_freq
            largest_clust_dat[i,1] = max(counts) 
        
        
        ##### writing output to file #####
        
        opfile=open(larg_clust_name,"w")
        for i in range(num_frames):
            opfile.write("%20.8f\t%20.15f\n"
            %(largest_clust_dat[i,0],largest_clust_dat[i,1]))
        opfile.close()
        
        opfile=open(num_clust_name,"w")
        for i in range(num_frames):
            opfile.write("%20.8f\t%20.15f\n"
            %(num_clust_dat[i,0],num_clust_dat[i,1]))
        opfile.close()





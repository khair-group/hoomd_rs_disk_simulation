#!/usr/bin/python3

import numpy as np
#from StringIO import StringIO
import sys
import math as m
import os
import re

l = len(sys.argv)

if l<4:
    print("\n Correct syntax is [calc_clust_stats.py] [path to input folder] [time_to_steady_state as fraction of total sim] [name of output file] \n")
    exit(0)

else :
    nlines = 0;


inp_dir = sys.argv[1]
t_s=(float(sys.argv[2]))
op_file_name = sys.argv[3]

full_path_inp_dir=os.getcwd() + "/" + inp_dir
directory = os.fsencode(full_path_inp_dir)

full_path_op_file = os.getcwd() + "/" + inp_dir + "/" + op_file_name


proc_dat=np.empty((0,3),float)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith("largest_clust") and filename.endswith(".dat"):
        basename, file_extension = os.path.splitext(filename)
        larg_clust_name = full_path_inp_dir + "/largest_clust_" + basename + ".dat"
        num_clust_name = full_path_inp_dir + "/num_clust_" + basename + ".dat"
        print(filename, " is the file I am reading")

        full_filename=full_path_inp_dir + "/" + filename

## programmatically extract the D_r and \omega from the file name
## idea taken from : https://stackoverflow.com/questions/4666973/how-to-extract-the-substring-between-two-markers
        try:
            d_r_found = re.search('D_r_(.+?)_omega', filename).group(1)
        except AttributeError:
            d_r_found = '' # apply your error handling

        try:
            omega_found = re.search('omega_(.+?)_tsteps', filename).group(1)
        except AttributeError:
            omega_found = '' # apply your error handling


        d_r=float(d_r_found)
        omega=float(omega_found)
        gamma=omega/d_r
#        print("D_r = ", d_r, "omega = ", omega, "gamma = ", gamma)

        f=open(full_filename,"r")
        arr=np.loadtxt(f)
        trim_len=int(np.ceil(t_s*len(arr)))
        rel_vals=arr[trim_len:,1]
        av_val=np.mean(rel_vals)
        std_val=np.std(rel_vals)
#        print(std_val)
        f.close()


        row_wise=np.array([gamma, av_val, std_val])
        proc_dat=np.vstack([proc_dat,row_wise])

# sorting output in increasing order of frequency
ind = np.argsort(proc_dat[:,0])
sorted_proc_dat = proc_dat[ind]

print(sorted_proc_dat)

#np.savetxt(op_file_name, proc_dat, fmt=['%.3f\t','%.3f\t', '%.3f'])

np.savetxt(full_path_op_file, sorted_proc_dat)




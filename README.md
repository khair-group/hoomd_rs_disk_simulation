# hoomd_rs_disk_simulation
HOOMD code to simulate a collection of RS disks

1. "box2d_create.py" creates a periodic square box and places particles inside it. This generated configuration is used as
    the input for HOOMD simulations.

2. "automate_mips_driver.py" accepts the configuration file geenrated by (1) and simulates a collection of RS disks whose self- 
    propulsion speeds oscillate periodically at a frequency $\omega$ between +0.1 and -0.1. The orientation of the disks evolve 
    according to a diffusive process, with a rotational diffusion constant $D_r$. This script performs the calculations for a 
    list of $\omega$ values that are specified within the code.

3. "calc_mean_std_clust_size.py" performs hierarachical clustering on the simulation output generated by (2), and returns a 
    timeseries of the size of the largest cluster and number of clusters.

4. "ret_clust_stats.py" returns time-averaged values for the mean and variance in cluster size.

Appropriate citation for HOOMD: J. A. Anderson, J. Glaser, and S. C. Glotzer. HOOMD-blue: A Python package for high-performance molecular dynamics and hard particle Monte Carlo simulations. Computational Materials Science 173: 109363, Feb 2020. 10.1016/j.commatsci.2019.109363


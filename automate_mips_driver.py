import multiprocessing as mp
import gsd.hoomd
import hoomd
import numpy as np
import os
import sys

l=len(sys.argv)

if l<9:
    print("\n Correct syntax is [automate_mips_driver.py] [path to input config file] [restart? (0/1)] [delta t] [no of timesteps] [D_r] [radius of disk] [output folder for results] [save_frequency] \n")
    exit(0)

else :
    pass


inp_file_path=sys.argv[1]
restart_param=int(float(sys.argv[2]))
time_step_width=float(sys.argv[3])       # timestep width used in simulation 
sim_length=int(float(sys.argv[4])) 	 # total number of timesteps for which simulation must be run
rot_diff_const=float(sys.argv[5])        # rotational diffusion constant
rad_disk=float(sys.argv[6])      	 # radius of disk
op_folder_path=sys.argv[7]               # path to input file, containing the raw, wrapped positions
save_freq=int(sys.argv[8])               # time-intervals at which the simulation output is written to file




class PrintTimestep(hoomd.custom.Action):

    def act(self, timestep):
        print(timestep, " timesteps computed")

# Calculate custom force in HOOMD
class CustomActiveForce_2D(hoomd.md.force.Custom):
 def __init__(self,f_array,freq,rotation_diff,N,dt):
   super().__init__(aniso=False)
   self.f_array = f_array #reshape into column vector
   self.rotation_diff = rotation_diff
   self.active_fi=np.zeros((N,3))
   self.freq=freq
   self.dt=dt
   self.N=N

 def update_force(self,timestep):

   with self._state.cpu_local_snapshot as data:
       quati = np.array(data.particles.orientation.copy())
       ptag = np.array(data.particles.tag.copy())


   rotation_constant = np.sqrt(2*self.rotation_diff*self.dt)
   temp_b=np.array([0.,0.,1.])  
   b=np.zeros((self.N,3))
   b[:]=temp_b	

   delta_theta=np.random.normal(0,1,[self.N,1])*(rotation_constant)
   sin_comp=np.multiply(b,np.sin(delta_theta/2.))
   cos_comp=np.cos(delta_theta/2.)

   q_scal = (quati[:,0]).reshape(self.N,1)
   q_vec = (quati[:,1:4]).reshape(self.N,3)

## See lines 853-856 in hoomd-blue/hoomd/VectorMath.h available on the gitHub page: 
## https://github.com/glotzerlab/hoomd-blue/blob/8fc5b26a4a28d42c86afc3740437aea66578af19/hoomd/VectorMath.h#L713
   scal_comp = cos_comp*q_scal - ((np.sum(sin_comp*q_vec,axis=1)).reshape(self.N,1))
## the second term on the RHS of the above equation implements row-wise dot-product
   vec_comp = np.multiply(cos_comp,q_vec) + np.multiply(q_scal,sin_comp) + np.cross(sin_comp,q_vec,axisa=1,axisb=1) 

   quati = np.concatenate((scal_comp,vec_comp),axis=1)

   q_scal = (quati[:,0]).reshape(self.N,1) #update after rotation
   q_vec = (quati[:,1:4]).reshape(self.N,3) #update after rotation

### end of rotational diffusion implementation. Now starting force calculation

   f = self.f_array*np.cos(self.freq*timestep*self.dt)

### rotating f by the quaternion
## See lines 947-960 in hoomd-blue/hoomd/VectorMath.h available on the gitHub page:
## https://github.com/glotzerlab/hoomd-blue/blob/8fc5b26a4a28d42c86afc3740437aea66578af19/hoomd/VectorMath.h#L947

   vec_term1 = (q_scal*q_scal - ((np.sum(q_vec*q_vec,axis=1)).reshape(self.N,1)))*f
   vec_term2 = 2*q_scal*np.cross(q_vec,f)
   vec_term3 = ((2*np.sum(q_vec*f,axis=1)).reshape(self.N,1))*q_vec

   self.active_fi = vec_term1 +vec_term2 + vec_term3
       

### update orientations ####
   with self._state.cpu_local_snapshot as data:
       data.particles.orientation = quati


 def set_forces(self,timestep):
   self.update_force(timestep)
   with self.cpu_local_force_arrays as arrays:
     arrays.force[:] =self.active_fi
     pass



# Function to build and run simulation
def simulate(fname,op_dir,w):

    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    if(restart_param==0):
        sim.timestep=0

    sim.create_state_from_gsd(fname)
 

    N = sim.state.N_particles
    # More simulation parameters
    tsteps=sim_length # total number of simulations steps to be executed
    dt = time_step_width # timestep width used for integration

    integrator = hoomd.md.Integrator(dt, integrate_rotational_dof=False)
    cell = hoomd.md.nlist.Cell(buffer=0.4)

    f_a=0.1 # strength of active force
    f_array =  f_a*np.ones((N,3))
    f_array[:,1:3]=0.

    custom_active =CustomActiveForce_2D(f_array = f_array, freq=w, rotation_diff=d_r, N=N, dt=dt)
    integrator.forces.append(custom_active)

    # Heyes-Melrose custom excluded volume interaction
    gamma = 1
    rad = rad_disk
    r_hs_min = rad
    r_hs_cut = 2.*rad
    size_hs = 2.*rad
    r = np.linspace(r_hs_min, r_hs_cut, 2, endpoint=False)

    # Create hard sphere interaction potential and force
    U_hs = gamma/(4*dt)*(r-2*rad)**2
    F_hs = -gamma/(2*dt)*(r-2*rad)
    # Use tabulated potential for hard sphere interactions
    hard_sphere = hoomd.md.pair.Table(nlist=cell, default_r_cut=r_hs_cut)

    hard_sphere.params[('A', 'A')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)

    if (hs_flag>0):
        integrator.forces.append(hard_sphere)

    # Use overdamped-viscous integrator
    odv = hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    odv.gamma.default = gamma
    odv.gamma_r.default = [0, 0, 0]  # Ignore rotational drag

    integrator.methods.append(odv)
    sim.operations.integrator = integrator

    # Run simulation
    custom_action = PrintTimestep()
    custom_op = hoomd.write.CustomWriter(
    action=custom_action,
    trigger=hoomd.trigger.Periodic(1000))
    sim.operations.writers.append(custom_op)


    op_file_name='N_%s_rs_hs_flag_%s_D_r_%s_omega_%f_tsteps_%s_dt_%s_phantom_abp.gsd' %(N,hs_flag,d_r,w,tsteps,dt)
    gsd_writer = hoomd.write.GSD(
        filename= op_dir + op_file_name,
        trigger=hoomd.trigger.Periodic(save_freq),
        mode='wb')
    sim.operations.writers.append(gsd_writer)
    sim.run(tsteps)

# Set global variables
inp_config=inp_file_path #input configuration file
hs_flag=1 #whether or not to include steric repulsion, 0 means phantom disks
d_r=rot_diff_const # rotational diffusion constant

#omega_list=np.array([0.1*d_r,0.5*d_r,1.*d_r,5.*d_r,10.*d_r,50.*d_r,100.*d_r,200.*d_r,300.*d_r,400.*d_r,500.*d_r,1000.*d_r,5000.*d_r]) #angular frequency of speed update

#omega_list=np.array([100.*d_r,200.*d_r,300.*d_r,1000.*d_r,5000.*d_r]) #angular frequency of speed update

omega_list=np.array([0.1*d_r,0.5*d_r,1.*d_r,5.*d_r,10.*d_r,50.*d_r,400.*d_r,500.*d_r]) #angular frequency of speed update


# Set up simulation and write to the appropriate directory

curdir = os.getcwd()
resdir=op_folder_path

try:
    os.mkdir(resdir)
    print("Directory made")
except FileExistsError:
    print("Directory already exists")


# Run simulations for all initial states
#p = mp.Pool()
#p.map(simulate, flist)


for i,w in enumerate(omega_list):
    print ("Now running \omega = ",w)
    simulate(inp_config,resdir,w)


prompt_file_name=resdir + "input_prompt.txt"
p_file=open(prompt_file_name,"w")
p_file.write("Input syntax: [automate_mips_driver.py] [path to input config file] [delta t] [no of timesteps] [D_r] [radius of disk] [output folder for results] [save_frequency]")
p_file.write("\n")
p_file.write("Actual input: " + str(sys.argv))
p_file.close()



print("Simulations completed")

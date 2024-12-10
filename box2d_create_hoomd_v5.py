#### Code taken from tutorial : https://hoomd-blue.readthedocs.io/en/v3.11.0/tutorial/00-Introducing-HOOMD-blue/03-Initializing-the-System-State.html#



import itertools
import math

import hoomd
import numpy

import gsd.hoomd

#m = 10
#N_particles = 4 * m**2

N_particles = 1600



spacing = 1.1
K = math.ceil(N_particles**(1 / 2))
#L = K * spacing
L = 100

x = numpy.linspace(-L/2., L/2., K, endpoint=False)
position = list(itertools.product(x, repeat=2))

for i in range(len(position)):
    position[i]=(position[i][0],position[i][1],0.)
#    print(position[i])

frame = gsd.hoomd.Frame()
frame.configuration.box = [L, L, 0, 0, 0, 0]
frame.particles.N = N_particles
frame.particles.position = position[0:N_particles]
frame.particles.typeid = [0] * N_particles

frame.particles.types = ['A']


op_file_name='N%s_L%s.gsd' %(N_particles,L)

with gsd.hoomd.open(name=op_file_name, mode='w') as f:
    f.append(frame)


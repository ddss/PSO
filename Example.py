"""
Example of main program to be used with PSO
@author: Renilton
"""
from PSO import PSO  # Importing the PSO class
from Graphics import Graph  # Importing the Graphics
from Modelo_benchmark import modelo_benchmark  # Importing the function to be minimized
from numpy import array, shape, argmax, argmin

function = 'Shaffer'  # Ackley, Exponencial, Negative-exponencial, Rastrigin, Rosenbrook, Shaffer
# pso_version =  # SPSO, PSO-WL, PSO-WR pso_chongpeng
bounds =  array([[-2, 2], [-2, 2]]) # input limits [(x_min,x_max)]
#number_particles =   # amount of particles
# interactions = 500  # amount of interactions that each particle will make
pso = PSO(lambda x: modelo_benchmark(x, function, bounds),
          pso_version='PSO-WL',
          bounds = bounds,
          num_part=10,
          maxiter=50,
          c1=float(1),  # cognitive constant
          c2=float(2),  # social constant
          wi=float(0.9),  # initial inertia
          wf=float(0.4),  # final inertia
          initial_swarm=None,  # Pattern, Sobol or None
          vel_restraint=float(-0.01),  # velocity restraint
          sig_evolution_value=float(1e-6),  # significant evolution
          significant_evolution=int(1000))  # stopping criterion

#pso.map_region2()
pso.minimize()
print(pso.gbest)
print(pso.fit_gbest)

# graph = Graph(pso.history._position, pso.history._fitness, pso.history._velocity, bounds, pso.gbest, pso.fit_gbest, pso.inter, number_particles)
# graph.pos_fit_3d()
# graph.positions()
# graph.pos_fit_2d()
# graph.int_fitness()
# graph.int_fit_average()
# graph.int_fit_sd()
# graph.int_velocity()
# graph.int_vel_average()
# graph.int_vel_sd()

# REGION
pso.history.region(1.0, 'down')
graph = Graph(pso.history.position_region, pso.history.fitness_region, pso.history._velocity, bounds, pso.gbest, pso.fit_gbest, pso.history.inter, pso.num_part)
graph.pos_fit_3d()
# graph.positions()
# graph.pos_fit_2d()

# MAPPING
pso.map_region2()
pso.history.region(1.0, 'down')
graph2 = Graph(pso.history.position_region, pso.history.fitness_region, pso.history._velocity, bounds, pso.gbest, pso.fit_gbest, pso.history.inter, pso.num_part)
graph2.pos_fit_3d()
# graph.positions()
# graph.pos_fit_2d()

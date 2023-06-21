"""
Example of main program to be used with PSO
@author: Renilton
"""
from PSO import PSO  # Importing the PSO class
from Graphics import Graph_cut, Graph  # Importing the Graphics
from Modelo_benchmark import modelo_benchmark  # Importing the function to be minimized
from numpy import array, shape, argmax, argmin

function = 'Rastrigin'  # Ackley, Exponencial, Negative-exponencial, Rastrigin, Rosenbrook, Shaffer
bounds = array([[-10, 10], [-10, 10]])
pso = PSO(lambda x: modelo_benchmark(x, function, bounds),
          pso_version='PSO-WL',  # SPSO, PSO-WL, PSO-WR pso_chongpeng
          bounds=bounds,  # input limits [(x_min,x_max)]
          num_part=30,  # amount of particles
          maxiter=500,  # amount of interactions that each particle will make
          c1=float(1),  # cognitive constant
          c2=float(2),  # social constant
          wi=float(0.9),  # initial inertia
          wf=float(0.4),  # final inertia
          initial_swarm=None,  # Pattern, Sobol or None
          vel_restraint=float(-0.01),  # velocity restraint
          sig_evolution_value=float(1e-6),  # significant evolution
          significant_evolution=int(1000))  # stopping criterion


pso.minimize()
print(pso.gbest)
print(pso.fit_gbest)
graph = Graph(pso.history._position, pso.history._fitness, pso.history._velocity, bounds, pso.gbest, pso.fit_gbest, pso.inter, pso.num_part)
graph.pos_fit_3d()
graph.pos_cov_area()
graph.positions()
graph.pos_fit_2d()
graph.int_fitness()
graph.int_fit_average()
graph.int_fit_sd()
graph.int_velocity()
graph.int_vel_average()
graph.int_vel_sd()

pso.history.region(50, 'down')
graph = Graph_cut(pso.history.position_region, pso.history.fitness_region, bounds, pso.gbest, pso.fit_gbest, pso.history.inter, pso.num_part)
graph.pos_fit_3d()
graph.pos_cov_area()
graph.positions()
graph.pos_fit_2d()

# pso.map_region_RD(500, 0.2)
pso.map_region_MBR(None, 100, True, 1)
graph = Graph(pso.history._position, pso.history._fitness, pso.history._velocity, bounds, pso.gbest, pso.fit_gbest, pso.inter, pso.num_part)
graph.pos_fit_3d()
graph.pos_cov_area()
graph.positions()
graph.pos_fit_2d()

pso.history.region(50, 'down')
graph = Graph_cut(pso.history.position_region, pso.history.fitness_region, bounds, pso.gbest, pso.fit_gbest, pso.history.inter, pso.num_part)
graph.pos_fit_3d()
graph.pos_cov_area()
graph.positions()
graph.pos_fit_2d()

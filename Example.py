"""
Example of main program to be used with PSO
@author: Renilton
"""
from PSO import PSO  # Importing the PSO class
from Graphics import Graph  # Importing the Graphics
from Modelo_benchmark import modelo_benchmark  # Importing the function to be minimized
from numpy import array, empty, where, full, shape, vstack

function = 'Shaffer'  # 1-Exponencial, 2-Rastrigin, 3-Shaffer, 4-Ackley
pso_version = 'PSO-WL'  # 1-SPSO 2-PSO-WL 3-PSO-WR [1] 4-pso_chongpeng [2]
bounds = array([[-10, 10], [-10, 10]])  # input limits [(x_min,x_max)]
number_particles = int(30)  # amount of particles
interactions = int(500)  # amount of interactions that each particle will make
pso = PSO(lambda x: modelo_benchmark(x, function, (shape(bounds))[1]), pso_version, bounds, number_particles, interactions,
          c1=float(2),  # cognitive constant
          c2=float(1),  # social constant
          wi=float(0.9),  # initial inertia
          wf=float(0.4),  # final inertia
          initial_swarm=None)  # sobol startup

#restriction_max = 1
try:
    restriction_max = float(input("Please enter a restriction for the fit: "))
except ValueError:
    print("Oops! That was no valid number. Try again...")
print(pso.gbest)
print(pso.fit_gbest)
position = array([pso.history.position[0]])
velocity = array([pso.history.velocity[0]])
vel_average = array([pso.history.vel_average[0]])
for i in range(1,(shape(bounds))[0]):
    position = vstack((position, pso.history.position[i]))
    velocity = vstack((velocity, pso.history.velocity[i]))
    vel_average = vstack((vel_average, pso.history.vel_average[i]))
fit = pso.history.fitness

restriction = True if restriction_max is not None else False
if restriction == True:
    fmax = full(shape(fit), restriction_max)
    teste = fit > fmax
    fit[teste] = None

# limite inferior
# teste = pso.swarm[0].position<bounds[:,0]
# pso.swarm[0].position[where(teste)[0]]=bounds[:,0][where(teste)[0]]

graph = Graph(position, fit, velocity, vel_average, (shape(bounds))[0])
#graph.bokeh()
graph.plotly()
graph.matplotlib()

# References
# 1 - SANTANA, D. D. Inferência Estatística Clássica com Enxame de Partículas, Salvador: UFBA, 2014, p. 28–31.
# 2 - Huang Chongpeng, Zhang Yuling, Jiang Dingguo and Xu Baoguo, "On Some Non-linear Decreasing Inertia Weight Strategies in Particle Swarm Optimization*," in Proceedings of the 26th Chinese Control Conference, Zhangjiajie, Hunan, China, 2007, pp. 570-753.

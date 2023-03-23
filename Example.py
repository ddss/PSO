"""
Example of main program to be used with PSO
@author: Renilton
"""
from PSO import PSO  # Importing the PSO class
from Graphics import Graph  # Importing the Graphics
from Modelo_benchmark import modelo_benchmark  # Importing the function to be minimized
from numpy import array, where, full

function = 2  # 1-Exponencial, 2-Rastrigin, 3-Shaffer, 4-Ackley
pso_version = 'PSO-WL'  # 1-SPSO 2-PSO-WL 3-PSO-WR [1] 4-pso_chongpeng [2]
bounds = array([[-2, 2], [-2, 2]])  # input limits [(x_min,x_max)]
number_particles = int(30)  # amount of particles
interactions = int(500)  # amount of interactions that each particle will make
pso = PSO(lambda x: modelo_benchmark(x, function, len(bounds)), pso_version, bounds, number_particles, interactions,
          c1=float(2),  # cognitive constant
          c2=float(1),  # social constant
          wi=float(0.9),  # initial inertia
          wf=float(0.4),  # final inertia
          initial_swarm='Sobol')  # sobol startup

restriction_max = 0.5
restriction_min = -0.5
print(pso.gbest)
print(pso.fit_gbest)


# RESTRIÇÕES!!!!
# x = array([1., 2., 3., 4., 5., 6.])
# xmax = array([0.5, 0.5, 0.5, 10, 10, 10])
# teste = x > xmax
# x[x > xmax] = xmax[where(teste)[0]]
pos = pso.history.position
fit = pso.history.fitness
xmax = full((len(pos), len(bounds)), restriction_max)
xmin = full((len(pos), len(bounds)), restriction_min)
for i in range(0, len(bounds)):
    fit[pos[:, i] > xmax[:, i]] = None
    fit[pos[:, i] < xmin[:, i]] = None
pos[pos > xmax] = None
pos[pos < xmin] = None

graph = Graph(pos[1:, :], fit[1:, 0], pso.history.velocity[:, 0], pso.history.vel_average[:, 0])
graph.bokeh()
graph.plotly()
graph.matplotlib(len(bounds))

# References
# 1 - SANTANA, D. D. Inferência Estatística Clássica com Enxame de Partículas, Salvador: UFBA, 2014, p. 28–31.
# 2 - Huang Chongpeng, Zhang Yuling, Jiang Dingguo and Xu Baoguo, "On Some Non-linear Decreasing Inertia Weight Strategies in Particle Swarm Optimization*," in Proceedings of the 26th Chinese Control Conference, Zhangjiajie, Hunan, China, 2007, pp. 570-753.

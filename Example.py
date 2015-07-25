# -*- coding: utf-8 -*-
"""
Example of main program to be used with PSO
@author: Daniel
"""
from matplotlib import use
use('Agg')

from PSO import PSO # Importing the PSO class
from Modelo_benchmark import Modelo # Importing the function to be minimized
from time import time

t1 = time()

sup = [100,100] # maximum value for the parameters
inf = [-100,-100] # minimum value for the parameters

args_model=['ackley'] # In this example, the Modelo class needs another argument: the name of the function to be minimized
# PSO Algorithm executed with decreasing inercia weight from 0.9 to 0.4; acceleration factors constant and equal to default values of 2; 30 particles and 200 iterations.
Otimizacao = PSO(sup,inf,{'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','restricao':True,'parada':['evolucaogbest','desviorelativo']},\
Num_particulas=30,itmax=500,w=[0.9,0.4],args_model=args_model,NP=2)
# Otimizacao = PSO(sup,inf,{'busca':'Otimo','algoritmo':'HPSO','gbest':'Particula','parada':'itmax','inercia':'Constante'},\
# Num_particulas=30,itmax=100,w=[0.9,0.4],args_model=args_model)
Otimizacao.Busca(Modelo,printit=True) # Do the search
Otimizacao.Relatorios(resumos_txt=True) # Printing results in the same folder as txt files
Otimizacao.Graficos(Nome_param=[r'$\theta_{}$'.format(i) for i in xrange(len(sup))],FO2a2=True) # Creating performance graphs in the same folder, including the objective function in 3d (FO2a2=True)
# To create movie frames uncomment the line below
Otimizacao.Movie(Nome_param=[r'$\theta_1$',r'$\theta_2$'],Unid_param=['admin','admin']) # Creating movie frames

# Printing the results:
print 'Optimum point', Otimizacao.gbest # optimum point
print 'Best Fitness' , Otimizacao.best_fitness # objective function in the optimum point
print 'Time', time()-t1
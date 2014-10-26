# -*- coding: utf-8 -*-
"""
Example of main program to use with PSO
@author: Daniel
"""

from PSO import PSO
from Modelo_benchmark import Modelo

sup = [32,32]
inf = [-32,-32]

Otimizacao = PSO(sup,inf,{'busca':'Otimo','algoritmo':'PSO','inercia':'Constante','aceleracao':'Constante','restricao':True},30,200,w=[1.0],args_model=['ackley'])
Otimizacao.Busca(Modelo)
Otimizacao.Result_txt()
Otimizacao.Movie(Nome_param=[r'$\theta_1$',r'$\theta_2$'],tipos=['projecao','funcao'])
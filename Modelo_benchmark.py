# -*- coding: utf-8 -*-
"""
Example of a program containg a function to be minimized
@author: Daniel
"""
from threading import Thread
import sys
from numpy import exp, size, sin, cos, sqrt, pi, sum, matrix

class Modelo(Thread):
    result = 0
    def __init__(self,param,args_model):
        Thread.__init__(self)
        self.args = param
        self.args_model = args_model
    
    def run(self):
	
	#Funções unimodais
	
	if self.args_model[0] == 'exponential':
	    # Função Exponencial
	    aux = -exp(-0.5*sum([arg**2 for arg in self.args]))
	
	if self.args_model[0] == 'sphere':
	    # Função sphere (f2)
	    aux = sum([arg**2 for arg in self.args])
	
	if self.args_model[0] == 'rosenbrook':
	    # Função de Rosenbrock
	    aux = 0
	    for i in xrange(0,size(self.args)-1):
	        aux = (100*(self.args[i+1]-self.args[i]**2)**2 + (self.args[i]-1)**2) + aux
	
	#Funções multimodais
	
	if self.args_model[0] == 'shaffer_f6':
	    # Shaffer's f6
	    num = sin(sqrt(self.args[0]**2 + self.args[1]**2))**2 - 0.5
	    den = (1.0 + 0.001*(self.args[0]**2 + self.args[1]**2))**2
	    aux = 0.5 + num/den
	
	if self.args_model[0] == 'rastrigin':
	    # Rastrigin
	    aux = sum([arg**2 - 10*cos(2*pi*arg) + 10 for arg in self.args])
 	
	if self.args_model[0] == 'ackley':
	    # Ackley
	    D = float(size(self.args))
	    aux = -20*exp(-0.2*sqrt((1/D)*sum([arg**2 for arg in self.args]))) -exp((1/D)*sum(cos([2*pi*arg for arg in self.args]))) + 20 + exp(1)
	
	self.result = aux

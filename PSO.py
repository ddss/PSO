# -*- coding: utf-8 -*-
"""
@title : Particle Swarm Optimization (PSO)
@author: Daniel Diniz Santana
@group: GI-UFBA (Grupo de Pesquisa em Incerteza da UFBA) (www.gi.ufba.br)

Algortimos de otimização suportados (Optimization Algorithms supported):
- PSO                 - Particle swarm optimization
- HPSO                - Self-Organizing Hierarchical Particle Swarm Optimizer

Fatores de inércia suportados (Inertia weight supported):
- Constante           
- TVIW-linear         - time variating inertia weight (linear)
- TVIW-random         - time variating inertia weight (random)
- TVIW-Adaptative-VI  - time variating inertia weight with adaptive parameter tuning of particle swarm optimization based on velocity information 

Fatores de aceleração suportados (Acceleration factors supported):
- Constante
- TVAC                - time variating acceleration coefficientes (linear)

- Constriction factor method:
- CFM                 - constriction factor method

Reinitialization velocity for HPSO:
- TVVr-linear         - time variating reinitialization velocity (linear)
- Constante
"""

#-------------------------------------------------------------------------------
# IMPORTAÇÃO DE PACOTES (packages import)
#-------------------------------------------------------------------------------

# Importação de pacotes para cálculos
import os
from time import sleep, ctime, time
import sys
from threading import Thread, Lock, BoundedSemaphore

from numpy import random, min, argmin, copy, matrix, multiply, random, size, min,\
max, copysign, ones, mean, std, sqrt, cos, pi, sort, argsort, savetxt
from scipy.misc import factorial
from math import isnan
import codecs

# Importação de pacotes para gráficos
from matplotlib import use, cm, ticker
use('Agg')


from matplotlib.pyplot import figure, axes, plot, subplot, xlabel, ylabel,\
    title, legend, savefig,  xlim, ylim, close, gca, hist
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import Ellipse, FancyArrowPatch

#-------------------------------------------------------------------------------
# Subrotinas e classes auxiliares
#-------------------------------------------------------------------------------

def Validacao_Diretorio(diretorio):
    # Basedo em códigos de terceiros (Based on interne: 
    # Validação da existência de diretório
    directory = os.path.split(diretorio+"Teste.txt")[0]

    if directory == '':
        directory = '.'

    # Se o diretório não existir, crie
    if not os.path.exists(directory):
        os.makedirs(directory)


class Arrow3D(FancyArrowPatch):
# Seta em gráfico 3D
# http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
		
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#-------------------------------------------------------------------------------
# ÍNICIO (beginning)
#-------------------------------------------------------------------------------

class Particula(Thread):
    # Classe que define a partícula, cada partícula é definida como uma Thread
    velocidade = 0
    fitness    = 0
    posicao    = 0
    Vo         = 0
    phi        = 0
    qsi        = 0
    pbest      = 0
    
    def __init__ (self,FO,w,C1,C2,k,Vmax,Vreinit,it,num_parametros,num_particulas,metodo,limite_superior,limite_inferior,ID_particle,args_model=[]):

        global vetor_posicoes, vetor_fitness, vetor_velocidades
        
        Thread.__init__(self)
        
        # Atribuição da Função objetivo
        self.FO = FO

        # Atribuição de variáveis self
        self.metodo   = metodo
        self.__w      = w
        self.__C1     = C1
        self.__C2     = C2
        self.__k      = k
        self.Vmax     = Vmax
        self.V_reinit = Vreinit
        self.__it     = it
        self.__args_model = args_model
        
        self.__vetor_posicoes    = vetor_posicoes
        self.__vetor_fitness     = vetor_fitness
        self.__vetor_velocidades = vetor_velocidades
        self.__num_particulas    = num_particulas
        self.__num_parametros    = num_parametros
        
        self.ID_particle = ID_particle
        
        #Limites superiores
        self.__limite_superior = limite_superior
        self.__limite_inferior = limite_inferior
        
        
        # Determinação da posição atual da partícula, velocidade inicial (PSO) e pbest 
        
        self.posicao  = copy(self.__vetor_posicoes[self.__it-1][self.ID_particle])    # Posição da partícula
        self.Vo       = copy(self.__vetor_velocidades[self.__it-1][self.ID_particle]) # Velocidade na iteração anterior
        self.posicao  = self.posicao.tolist()
        # Correção de self.posicao e self.Vo
        if isinstance(self.Vo[0].tolist(), list):
            self.Vo       = self.Vo[0].tolist()
        if isinstance(self.posicao[0], list):
            self.posicao = self.posicao[0].tolist()

        # Busca pela melhor posição desta partícula:
        vetor_fitness_particula  = []
        vetor_posicoes_particula = []
        
        for i in xrange(1,self.__it+1):
            vetor_fitness_particula.append(self.__vetor_fitness[i-1][self.ID_particle])
            vetor_posicoes_particula.append(self.__vetor_posicoes[i-1][self.ID_particle])
        
        best_position = argmin(vetor_fitness_particula)
        
        self.pbest = vetor_posicoes_particula[best_position] # Posição para a qual a partícula obteve o melhor fitness
    
    def Fitness_Particula(self):
        # Cálculo do Fitness da partícula
        Thfitness = self.FO(self.posicao,self.__args_model)
        Thfitness.start()
        Thfitness.join()

        self.fitness = Thfitness.result
    
    def Execucao_PSO(self):
        # Método para cálculo do particle swarm optimization (PSO) e suas variações
        global gbest
        
        # Cálculo da velocidade
        R1 = random.uniform(0,1,size(self.posicao)) # Termo de aceleração aleatório
        R2 = random.uniform(0,1,size(self.posicao)) # Termo de aceleração aleatório
    
        #sys.stdout.write('Vo '+ str(self.Vo)+'w '+ str(self.__w)+ 'pos '+ str(self.posicao)+ 'pbest '+ str(self.pbest)+ 'gbest '+ str(gbest)+'\n')
        #sys.stdout.flush()
        vel1 = [Vo*(self.__w) for Vo in self.Vo ]
        vel2 = [self.__C1*R1[j]*(self.pbest[j]-self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel3 = [self.__C2*R2[j]*(gbest[j] - self.posicao[j])    for j in xrange(self.__num_parametros)]

        if self.metodo.CFM == False:
            vel4 = [vel1[j]+vel2[j]+vel3[j] for j in xrange(self.__num_parametros)]
        else:
            self.phi = self.__C1+self.__C2

            if self.phi > 4.0:
                self.qsi = 2*self.__k/(self.phi-2+sqrt(self.phi**2-4.0*self.phi))
            else:
                self.qsi = self.__k

            vel4 = [self.qsi*(vel1[j]+vel2[j]+vel3[j]) for j in xrange(self.__num_parametros)]

	if self.metodo.inercia != 'TVIW-Adaptative-VI':
	    # Verificação se a velocidade calculada é menor que Vmax, para cada dimensão
	    self.velocidade = [0]*self.__num_parametros

            for i in xrange(self.__num_parametros):
                vel5 = min((abs(vel4[i]),self.Vmax[i]))
                aux = copysign(vel5,vel4[i])
                if not isinstance(aux,float):
                    self.velocidade[i] = float(aux)
                else:
                    self.velocidade[i] = aux

	else:
	    self.velocidade = vel4

        # Nova posição
        self.posicao    = [self.posicao[j]+self.velocidade[j] for j in xrange(self.__num_parametros)]
    
        # Validação de o valor da posição está conttido dentro da região da restrição em caixa (limites originais do problema)
        if self.metodo.restricao == True:
            for i in xrange(self.__num_parametros):
                if self.posicao[i] > self.__limite_superior[i]:
                    self.posicao[i]       = self.__limite_superior[i]
                    self.velocidade[i] = -0.10*self.velocidade[i]
                    
                elif self.posicao[i] < self.__limite_inferior[i]:
                    self.posicao[i]       = self.__limite_inferior[i]
                    self.velocidade[i] = -0.10*self.velocidade[i]
                    
        self.Fitness_Particula()
       
    def Execucao_HPSO(self):
        global gbest
        
        # Cálculo da velocidade

        R1 = random.uniform(0,1,size(self.posicao)) # Termo de aceleração aleatório
        R2 = random.uniform(0,1,size(self.posicao)) # Termo de aceleração aleatório
    
        vel1 = [self.__C1*R1[j]*(self.pbest[j]-self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel2 = [self.__C2*R2[j]*(gbest[j] - self.posicao[j])    for j in xrange(self.__num_parametros)]                    
        vel3 = [vel1[j]+vel2[j]                                 for j in xrange(self.__num_parametros)]    
    
        self.velocidade = [0]*self.__num_parametros
        for i in xrange(self.__num_parametros):
                
            if vel3[i] == 0:
                if random.uniform(0,1) < 0.5:
                    vel3[i] = random.uniform(0,1)*self.V_reinit[i]
                else:
                    vel3[i] = -random.uniform(0,1)*self.V_reinit[i]
                
            vel4 = min((abs(vel3[i]),self.Vmax[i]))
                
            self.velocidade[i] = copysign(vel4,vel3[i])
        
        # Nova posição
        self.posicao    = [self.posicao[j]+self.velocidade[j] for j in xrange(self.__num_parametros)]    
        
        # Validação de o valor da posição está conttido dentro da região da restrição em caixa (limites originais do problema)
        if self.metodo.restricao == True:
            for i in xrange(self.__num_parametros):
                if self.posicao[i] > self.__limite_superior[i]:
                    self.posicao[i] = self.__limite_superior[i]
                elif self.posicao[i] < self.__limite_inferior[i]:
                    self.posicao[i] = self.__limite_inferior[i]
            
            
        self.Fitness_Particula()
    
    def Armazenamento(self):
        
        global vetor_posicoes, vetor_velocidades, vetor_fitness, best_fitness, gbest, Controle_variaveis
        
        # Método para verificar se haverá mudança em gbest

        # Início da Região Crítica - Compartilhada por todas as threads
        Controle_variaveis.acquire()
        
        # Atualização dos vetores (históricos / Memória)
        vetor_posicoes[self.__it][self.ID_particle]    = self.posicao
        vetor_velocidades[self.__it][self.ID_particle] = self.velocidade
        vetor_fitness[self.__it][self.ID_particle]     = self.fitness
        
        # Teste para verificar mudança no mínimo global:
        if self.metodo.gbest == 'Particula':
            if self.fitness < best_fitness:
                    best_fitness  = self.fitness
                    if self.metodo.busca == 'Otimo':
                        gbest = self.posicao # O valor de gbest é apenas atualizado, caso esteja se buscando o ponto ótimo
        
        Controle_variaveis.release()
        # Fim da região crítica - Compartilhada por todas as threads
              
    def run(self):
        # Método obrigatório, visto que Particula é uma Thread
        
        global Controle_Particula, Controle_Iteracao, total_particulas_atendidas, Controle_Total_Threads
        
        # Escolha do tipo de algoritmo a ser executado
        if self.metodo.algoritmo == 'PSO':
            self.Execucao_PSO()
        elif self.metodo.algoritmo == 'HPSO':
            self.Execucao_HPSO()
        
        self.Armazenamento()
        
        # Controle de Threads
        Controle_Total_Threads.acquire()
        
        total_particulas_atendidas = total_particulas_atendidas+1

        if (total_particulas_atendidas)==self.__num_particulas:
            Controle_Iteracao.release()
        
        Controle_Total_Threads.release()
        
        Controle_Particula.release()

class Metodo:
    
    def __init__(self,metodo):
        '''
        Classe para transformar o argumento ``metodo`` do PSO, em atributos. Além de definir os valores default dos métodos.
        
        Chaves disponíveis: ``busca`` , ``algoritmo`` , ``inercia`` , ``aceleracao`` , ``CFM`` , ``Vreinit`` , ``restricao`` , ``gbest``
        
        =================================
        Conteúdos disponíveis por chaves:
        =================================
        
        **Chave obrigatória**
        
        * ``busca``: ``Otimo`` ou ``Regiao``
        
        **Chaves não obrigatórias** (com valores default)
        
        * ``algoritmo``: ``PSO`` ou ``HPSO``
            
        * ``inercia``: ``Constante`` ou ``TVIW-linear`` ou ``TVIW-Adaptative-VI`` ou ``TVIW-random``
        
        * ``aceleracao``: ``Constante`` ou ``TVAC``
        
        * ``CFM``: ``True`` ou ``False``

        * ``Vreinit``: ``Constante`` ou ``TVVr-linear``
        
        * ``restricao``: ``True`` ou ``False``

        * ``gbest``: ``Particula`` ou ``Enxame``
                
        ================
        Valores default:
        ================
        
        **Chaves obrigatórias**

        * ``busca``: define o método de busca. Conteúdos disponíveis são: ``Otimo`` ou ``Regiao``

        **Chaves não obrigatórias** 
        
        * ``algoritmo``: define o algoritmo a ser utilizado. Se ``busca`` for ``Regiao``, então é definido ``HPSO``. Caso contrário ``PSO``
            
        * ``inercia``: define o método do peso de inércia. Se ``algoritmo`` é ``PSO`` e ``busca`` è ``Regiao``, então, o método de inérci é ``TVIW-linear``. Caso contrário, ``Constante``
        
        * ``aceleracao``: define o método para fatores de aceleração. Se ``algoritmo`` é ``PSO``, ``aceleração`` é ``Constante``. Caso ``algoritmo`` seja ``HPSO`` e ``busca`` seja ``Otimo``, então ``TVAC``. Se ``busca`` for ``Regiao``, então ``Constante``.
        
        * ``CFM``: define se método CFM (constriction factor method) é utilizado. Caso não definido ele não é utilizado, pois conforme [11] ele pode ser entendido como um caso especial do método do peso de inércia com variação linear.
        
        * ``Vreinit``: método para definir o comportamento da reinicialização do HPSO. Caso não definido é utilizado o ``TVVr-linear``
        
        * ``restricao``: define se os limites serão utilizados como restrições. Caso não definido é assumido o valor lógico ``True``
        
        * ``gbest``: define como a atualização de ``gbest`` (ponto ótimo) é realizada. Se o método de busca é ``Otimo``, então ``Particula`` . Caso o método de busca seja ``Regiao``, então ``Enxame``.
        '''
        self.busca      = metodo.get('busca')
        self.algoritmo  = metodo.get('algoritmo')
        self.inercia    = metodo.get('inercia')
        self.aceleracao = metodo.get('aceleracao')
        self.CFM        = metodo.get('CFM')
	self.Vreinit    = metodo.get('Vreinit')
        self.restricao  = metodo.get('restricao')
        self.gbest      = metodo.get('gbest')
        
	for key in metodo.keys():
            if not key in ['busca','algoritmo','inercia','aceleracao','CFM','Vreinit','restricao','gbest']:
                raise NameError, u'Keyword: %s not available! Available keywords: busca, algoritmo, inercia, aceleracao, CFM, Vreinit, restricao, gbest. Please verify!'%(key)
    
    def Inicializacao(self):
        # Validação da existência dos métodos escolhidos

        if (self.busca != 'Otimo') and (self.busca != 'Regiao'):
            raise NameError, u'Métodos do PSO: Otimo e Regiao'
        
        if self.algoritmo == None:
            if self.busca == 'Regiao':
                self.algoritmo = 'HPSO'
            else:
                self.algoritmo = 'PSO'

        if (self.algoritmo != 'PSO') and (self.algoritmo != 'HPSO'):
            raise NameError, u'É suportado para o algoritmo PSO, HPSO'
        
        if self.CFM == None:
            self.CFM = False # CFM, conforme apresentado por [7] -> constriction factor method
                
        elif (self.CFM != True) and (self.CFM != False):
            raise NameError, u'CFM é suportado no método PSO: True ou False'
        
        elif ((self.CFM == True) or (self.CFM == False)) and (self.algoritmo == 'HPSO'):
            raise NameError, u'CFM é suportado no método PSO. Está sendo utilizado o HPSO. Utilzar None'    
        
        if self.inercia == None:
            if (self.algoritmo == 'PSO') and (self.busca=='Otimo'):
                if self.CFM == False:
                    self.inercia = 'TVIW-linear' #TVIW-linear, conforme apresentado por [1] e [4] -> standard particle swarm optimization
                else:
                    self.inercia = 'Constante'
            elif (self.busca == 'Regiao') and (self.algoritmo != 'HPSO'):
                self.inercia = 'Constante'
        elif (self.inercia != 'TVIW-linear') and (self.inercia != 'Constante') and (self.inercia != 'TVIW-random') and (self.inercia != 'TVIW-Adaptative-VI'):
            raise NameError, u'É apenas suportado o PSO com w: Constante ou TVIW-linear ou TVIW-random ou TVIW-Adaptative-VI'
        
        elif ((self.inercia == 'TVIW-linear') or (self.inercia == 'Constante') or (self.inercia == 'TVIW-random') or (self.inercia == 'TVIW-Adaptative-VI')) and  (self.algoritmo == 'HPSO'):
            raise NameError, u'O peso de inercia é apenas suportado do PSO. Está sendo utilizado HPSO utilizar None'
        
        elif (self.inercia == 'TVIW-Adaptative-VI') and (self.algoritmo != 'PSO'):
            raise NameError, u'O peso de inercia TVIW-Adaptative-VI é apenas suportado no PSO.'
        
        
        if self.aceleracao == None: # Método para cálculo de C1 e C2
            if (self.algoritmo == 'PSO'):
                self.aceleracao = 'Constante'
            elif (self.algoritmo == 'HPSO'):
                if self.busca == 'Otimo':
                    self.aceleracao = 'TVAC'
                elif self.busca == 'Regiao':
                    self.aceleracao = 'Constante'
        elif (self.aceleracao != 'Constante') and (self.aceleracao!= 'TVAC'):
            raise NameError, u'É suportado o PSO, HPSO, com aceleração (C1 e C2) constante ou usando TVAC'
        
        if self.algoritmo == 'HPSO':
            if self.Vreinit == None:
                self.Vreinit = 'TVVr-linear'
	    elif (self.Vreinit != 'TVVr-linear') and (self.Vreinit != 'Constante'):
		raise NameError, u'Use TVVr-linear or Constante for the Vreinit method'
	if (self.Vreinit != None) and (self.algoritmo != 'HPSO'):
	    raise NameError, u'Vreinit method is only applied with HPSO algorithm, please set Vreinit method to None'
	
        if self.restricao == None:
            self.restricao = True
        elif (self.restricao != True) and (self.restricao != False):
            raise NameError, u'A restrição assume valores lógicos True ou False'
        
        if self.gbest == None:
            if self.busca == 'Otimo':
                self.gbest = 'Particula'
            elif self.busca == 'Regiao':
                self.gbest = 'Enxame'
        elif (self.gbest != 'Particula') and (self.gbest != 'Enxame'):
            raise NameError, u'É suportado os métodos para gbest: Particula e Enxame'

class PSO:    

    def __init__(self,limite_superior, limite_inferior, metodo={'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'Constante','CFM':False,'restricao':True,'gbest':'Particula'}, Num_particulas=30, itmax=2000,**kwargs):
        """
        ************************************
        Partcle Swarm Optimization Algorithm 
        ************************************
        Algoritmo para otimização através do enxame de partículas, com suporte ao processamento paralelo rudimentar.

        As principais referências utilizadas são [1] a [11]
        
        =======================================================
        USO do algoritmo (Simplificado - Entradas obrigatórias)
        =======================================================
        
        **Entradas Obrigatórias**        
        
        * ``limite_superior``: é uma lista que define o limite superior de busca de todos as variáveis de decisão do problema
        * ``limite_inferior``: é uma lista que define o limite inferior de busca de todos as variáveis de decisão do problema
    
        O PSO define o número de parâmetros através da quantidade de termos presentes nestas listas, portanto eles devem ter o mesmo tamanho.        
        
        **Funçao objetivo**: a função a ser minimizada **DEVE** ser estruturada da seguinte forma: ::
            
            from threading import Thread
    
            class NomeFuncaoObjetivo(Thread):
                result = 0
                def __init__(self,parametros,argumentos_extras):
                    Thread.__init__(self)
                    self.param = p
                    self.args  = argumentos_extras
    
                def run(self):
                    
                    'Calculos utilizado self.param (e self.args, opcionalmente)'
                    
                    self.result =  'Resultado da função objetivo (deve ser um float)'
        
        A funçao objetivo deve possuir as seguintes características:
        
        * **Thread** : deve herdar a classe Thread (e *não* pode substituir o método __init__ da mesma) (ainda se avalia se isto é realmente necessário)
        * **método run**: este método é obrigatório por se tratar de uma Thread, o PSO o executará durante a otimizaçao
        * **atributo self.result**: atributo obrigatório, o PSO espera que a função objetivo tenha este atributo 
        * **entradas**: deve possuir obrigatoriamente 2 entradas: a primeira os parâmetros e a segunda os argumentos_extras. Ambos são passados como listas.
        
        ==========
        Exemplo 1
        ==========
        
        Deseja-se encontrar o ponto de mínimo da função: y = x^2. A variável de decisão é x
            
        * 1 passo: definir a função objetivo em um arquivo (Nome do arquivo aqui utilizado funcaoobjetivo.py) ::
        
            from threading import Thread
            
            class FO(Thread):
                result = 0
                def __init__(self,param,args):
                    Thread.__init__(self)
                    self.x = param
    
                def run(self):
                    
                    self.result =  self.x**2
            
        * 2 passo: realizar a otimização ::
            
                >>> from funcaoobjetivo import FO
                >>> sup  = [10.]  # limite superior de busca para x
                >>> inf  = [-10.] # limite inferir de busca para x
                >>> Otimizacao = PSO(sup,inf) # Criação da classe PSO
                >>> Otimizacao.Busca(FO)      # Comando para iniciar a busca
                >>> Otimizacao.Result_txt()   # Salvar os principais resultados em arquivos de texto
                >>> Otimizacao.Graficos()     # Criação de gráficos com indicadores de desempenho do algoritmo de PSO

        ==========
        Resultados
        ==========
        
        Na configuração apresentada  pelo Exemplo 1, o algoritmo irá criar 2 pastas no caminho onde está sendo executado: Resultados e Gráficos
        
        O objeto *Otimizacao* do exemplo 1 tem alguns atributos:
        
        * ``.gbest``: ponto ótimo encontrado pelo algoritmo
        * ``.best_fitness``: valor da função objetivo no ponto ótimo
        * ``.historico_posicoes``: histórico dos valores dos parâmetros a cada iteração, para cada partícula
        * ``.historico_fitness``: histórico dos valores da função objetivo a cada iteração, para cada partícula

        Para mais detalhes sobre os métodos ``.Busca()``, ``.Graficos()`` e ``.Result_txt()`` consulte a documentação dos mesmos.        
        
        =====================================
        USO do algoritmo (Entradas opcionais)
        =====================================
        
        Além de definir o limite_superior e limite_inferior, existem outras entradas que podem ser utilizadas
        
        * ``metodo``: deve ser um dicionário, indicando os métodos a serem utilizadas. As chaves (*strings*) e seus possíveis conteúdos (*strings* ou *bool*) são detalhadas abaixo:
        
            * ``busca`` (chave): define o objetivo da busca (**OBRIGATÓRIO**)

                * ``otimo``  (conteúdo, string): busca o ponto ótimo de uma função objetivo
                * ``regiao`` (conteúdo, string): avalia a região em torno de um ponto. (Este ponto deve ser entrado como a kwarg 'otimo')
            
            * ``algoritmo`` (chave): define o tipo de algoritmo a ser utilizado
                
                * ``PSO`` (Particle Swarm Optimization) (conteúdo, string): algoritmo de PSO com peso de inércia, fatores de aceleração, e suporte ao método CFM (constriction factor method)
                * ``HPSO`` (Self-Organizing Hierarchical Particle Swarm Optimizer) (conteúdo, string): executa um método de PSO assumindo peso de inércia zero e reiniciando as partículas quando suas velocidades são zero.
            
            * ``inercia``: define como o algoritmo calcula o peso de inércia
            
                * ``Constante`` (conteúdo, string): o peso de inércia é assumido constante e não varia ao longo das iterações (Vide [1] e [4])
                * ``TVIW-linear`` (*linear time varying inertia weight*) (conteúdo, string): o peso de inércia varia linearmente ao longo das iterações (VIde [4])
                * ``TVIW-Adaptative-VI`` (*adaptive time varying inertia weight based on velocity*) (conteúdo, string): o peso de inércia e ajustado no algoritmo para forçar o perfil de velocidade a um comportamento ideal (Vide [8])
                * ``TVIW-random`` (*random time varying inertia weight*) (conteúdo, string): o peso de inércia é variado aleatoriamente entre 0 e 1 (Vide [4])
            
            * ``aceleracao`` (chave):  define como o algoritmo calcula as constantes de aceleração
            
                * ``Constante`` (conteúdo, string):  os fatores de aceleração são assumidos constantes e não variam ao longo das iterações (Vide [4])
                * ``TVAC`` (*time varying acceleration constant*) (conteúdo, string):  os constantes de aceleraçao variam linearmente ao longo das iterações (Vide [10])
                
            * ``CFM`` (chave): define se o método utilizará o *constriction factor method* (Vide [5] e [6])
            
                * ``True`` (conteúdo, bool): o CFM é utilizado
                * ``False`` (conteúdo, bool): o CFM não é utilizado 
            
            * ``restriçao`` (chave): define se os limites de busca serão utilizados como restrições do problema de otimização
            
                * ``True`` (conteúdo, bool): os limites são utilizados como restrições
                * ``False`` (conteúdo, bool): os limites não são utilizados como restrições 
                
            * ``gbest`` (chave): define a forma como o ponto ótimo (gbest) é atualizado ao longo das iterações
            
                * ``Particula`` (conteúdo, string): gbest será atualizado ao longo da iteração pelas partículas
                * ``Enxame`` (conteúdo, string): gbest será atualizado ao final de cada iteração
            
            * ``Vreinit`` (chave):  define a forma de tratar a velocidade de reinicilização no algoritmo de HPSO (Vide [10])
            
                * ``TVVr-linear`` (tiem varying reinitialization velocity : a velocidade de reinicialização decresce linearmente ao longo das iterações
                * ``Constante`` a velocidade é mantida constante
                
        Na definição do método, apenas a chave ``busca`` é obrigatória. Se algumas das restantes não for definida, será assumido valores default. (vide a documentação da classe ``Metodo``)
        
        Caso o método seja omitido, na chamada da função PSO, conforme Exemplo 1, será utilizado o método: ::
        
        {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'Constante','CFM':False,'restricao':True,'gbest':'Particula'}
        
        **Sugestões e recomendações de métodos**:
        
            * PSO peso de inércia e fatores de aceleração constantes [1]: ::
        
                metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'Constante','aceleracao':'Constante','gbest':'Particula'}
            
            * PSO com descrescimento linear ao peso de inércia e fatores de aceleração constantes [3]: ::
        
                metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'Constante','gbest':'Particula'}
        
            * PSO com descrescimento linear do peso de inércia e fatores de aceleração [10]: ::
        
                metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','gbest':'Particula'}
        
            * PSO com peso de inércia aleatório e fatores de aceleração constantes [4]: ::
        
                metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-random','aceleracao':'Constante','gbest':'Particula'}
        
            * PSO com peso de inércia adaptativo e fatores de aceleração constantes [8]: ::
        
                metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-Adaptative-VI','aceleracao':'Constante','gbest':'Particula'}
                
            * HPSO com fatores de aceleração constantes e velocidade de reinicialização constante [10]: ::

                metodo = {'busca':'Otimo','algoritmo':'HPSO','inercia':None,'aceleracao':'Constante','gbest':'Particula'}

            * HPSO com fatores de aceleração e velocidade de reinicialização decrescendo linearmente [10]: ::

                metodo = {'busca':'Otimo','algoritmo':'HPSO','aceleracao':'Constante','Vreinit':'TVVr-linear','gbest':'Particula'}


        * **Num_particulas** (float): número de partículas utilizadas pelo algoritmo, normalmente entre 30 e 50. Caso não seja definido, será utilizado 30.
        
        * **itmax** (float): número máximo de iterações do algoritmo. Caso não seja definido, será utilizado 2000.

        ==========
        Exemplo 2
        ==========

        O **exemplo 1** será resolvido novamente, através do método de PSO com peso de inércia e fatores de aceleração variando linearmente. O algoritmo será realizado sem restrição. Devem ser realizadas 1000 iterações com 35 partículas ::
        
                >>> from funcaoobjetivo import FO
                >>> sup  = [10.]  # limite superior de busca para x
                >>> inf  = [-10.] # limite inferior de busca para x
                >>> metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','CFM':False,'restricao':False,'gbest':'Particula'}
                >>> Otimizacao = PSO(sup,inf,metodo,35,1000) # Criação da classe PSO
                >>> Otimizacao.Busca(FO)      # Comando para iniciar a busca
                >>> Otimizacao.Result_txt()   # Salvar os principais resultados em arquivos de texto
                >>> Otimizacao.Graficos()     # Criação de gráficos com indicadores de desempenho do algoritmo de PSO

        ========
        Keywargs
        ========    
        
        Outros argumentos opcionais estão disponíveis. Estes argumentos definem valores dos parâmetros do algoritmo a ser executado, em metodo. Por serem opcionais, caso não sejam definidos assumem valores *default*. Por serem **argumentos chave** devem ser expressamente definidos pelo nome.
        
        **Keywargs com valores default None** (Caso não definidas, são ignoradas pelo algoritmo)        
        
        * ``posinit_sup`` (lista de mesmo comprimento de ``limite_superior``): valores de **limite superior** de busca para inicializar o algoritmo em um intervalo diferente daquele definido pelo limite_superior 
        * ``posinit_inf`` (lista de mesmo comprimento de ``limite_inferior``): valores de **limite inferior** de busca para inicializar o algoritmo em um intervalo diferente daquele definido pelo limite_inferior 
        * ``otimo`` (lista): ponto focal quando o método busca utilizado é ``Regiao``. Deve ser uma lista com a mesma dimensão das dimensões de busca
        
        **keywargs cujos valores default dependem do método** (Caso não definidas, seus valores serão determinados pelo algoritmo) - vide seção valores default abaixo    

        * ``w`` (lista com 1 ou 2 elementos OU None): valores de peso de inércia. Caso o método escolhido para **inercia** seja constante, a lista deve conter apenas um elemento. Caso contrário, deve conter os valores iniciais e finais, respectivamente. Obs.: Para o algoritmo de ``HPSO`` ou com peso de inércia `TVIW-random``, ``w = None``
        * ``C1`` (lista com 1 ou 2 elementos): valores do fator de aceleração do coeficiente individual da velocidade. Caso o método escolhido para **aceleracao** seja constante, a lista deve conter apenas um elemento. Caso contrário, deve conter os valores iniciais e finais, respectivamente.           
        * ``C2`` (lista com 1 ou 2 elementos): valores do fator de aceleração do coeficiente social da velocidade. Caso o método escolhido para **aceleracao** seja constante, a lista deve conter apenas um elemento. Caso contrário, deve conter os valores iniciais e finais, respectivamente.
        * ``Vmax`` (lista com mesma dimensão do ``limite_superior`` e ``limite_inferior``): **velocidade máxima** das partículas para cda dimensão do problema, ou seja, para cada parâmetro.
        * ``Vreinit`` (lista com estrutura definida ou None): **velocidade de reinicialização** (usada apenas no algortimo de HPSO). A estrutura deve seguir ``[[Vreinit_1st_param,Vreinit_2nd_param,...,Vreinit_last_param],final_percentage]``. O primeiro argumento é uma lista com as velocidades de reinicializaçao iniciais para cada parâmetro. O segundo argumento (``final_percentage``) define as velocidades finais como uma porcetagem de ``Vmax``. Caso o método para ``Vreinit`` seja constante, ``final_percentage`` deve ser 1.0.
        
        **keywargs com valores default fixos**  (Caso não definidas, seus valores serão determinados pelo algoritmo)

        * ``deltaw``: parâmetro para de seleção para o cálculo de w, quando o método do peso de inércia é ``TVIW-Adaptative-VI``. Seu valor default é 0.1. Vide [8] para mais detalhes.
        * ``k``: parâmetro de ajuste quando o método CFM é utilizado. Seu valor default é 1. Vide [4] para detalhes.
        * ``gama``: parâmetro de ajuste quando o método CFM é utilizado. Seu valor default é 0.5. Vide [4] e [7]
        * ``args_model``: lista que possui argumentos extras a serem passados para a função objetivo. Seu valor default é uma lista vazia.

        ==========
        Exemplo 3
        ==========
        
        O **exemplo 2** será resolvido novamente, definindo os valores do peso de inércia (``w``) entre 0.9 e 0.3, fator de aceleração individual (``C1``) entre 2.5 e 0.5 e fator de aceleração social (``C2``) entre 0.5 e 2.5. Observe que ``w`` e ``C1`` decrescem, enquanto ``C22`` aumenta. ::
        
                >>> from funcaoobjetivo import FO
                >>> sup  = [10.]  # limite superior de busca para x
                >>> inf  = [-10.] # limite inferior de busca para x
                >>> metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','CFM':False,'restricao':False,'gbest':'Particula'}
                >>> w  = [0.9,0.3] # valores do peso de inércia
                >>> C1 = [2.5,0.5] # valores do fator individual de aceleração de aceleração 
                >>> C2 = [0.5,2.5] # valores do fator social de aceleração de aceleração 
                >>> Otimizacao = PSO(sup,inf,metodo,35,1000,w=w,C1=C1,C2=C2) # Criação da classe PSO
                >>> Otimizacao.Busca(FO)      # Comando para iniciar a busca
                >>> Otimizacao.Result_txt()   # Salvar os principais resultados em arquivos de texto
                >>> Otimizacao.Graficos()     # Criação de gráficos com indicadores de desempenho do algoritmo de PSO

        ================
        Valores default 
        ================

        Os valores default do algoritmo podem ser consultados nos seguintes caminhos:
        
        * Na documentação da classe ``Metodo`` (para o método)
        * Na documentação do método ``.default`` da classe PSO (Para o peso de inércia e fatores de aceleração)
        * Na documentação do método ``.inicializacao`` da classe PSO (Para as velocidades máxima e de reinicialização)
        
        **Observe que NEM todas as combinações de métodos possuem valores *default* de parâmetros. Use com cautela os valores default**

        =======
        Métodos
        =======
        
        * ``Busca``: executa a busca definida no argumento método
        * ``Result_txt``: salva arquivos de texto com os principais resultados
        * ``Graficos``: cria gráficos dos principais resultados
        * ``Movie``: cria gráficos das posições das partículas a cada iteração (pode levar muito tempo)

        ===========
        Referências
        ===========

        [1] KENNEDY, J.; EBERHART, R. Particle swarm optimization. Proceedings of ICNN’95 - International Conference on Neural Networks. Anais... [S.l.]: IEEE. , 1995

        [2] EBERHART, R.; KENNEDY, J. A new optimizer using particle swarm theory. MHS’95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science. Anais... [S.l.]: IEEE., 1995

        [3] SHI, Y.; EBERHART, R. A modified particle swarm optimizer. 1998 IEEE International Conference on Evolutionary. Computation Proceedings. IEEE World Congress on Computational Intelligence (Cat. No.98TH8360). Anais... [S.l.]: IEEE., 1998
        
        [4] EBERHART, R. C. Particle swarm optimization: developments, applications and resources. Proceedings of the 2001.Congress on Evolutionary Computation (IEEE Cat. No.01TH8546). Anais... [S.l.]: IEEE. , 2001
        
        [5] CLERC, M. The swarm and the queen: towards a deterministic and adaptive particle swarm optimization.
        
        [6] CLERC, M.; KENNEDY, J. The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, v. 6, n. 1, p. 58–73, 2002.
        
        [7] ZHANG, L.; YU, H.; HU, S. Optimal choice of parameters for particle swarm optimization. Journal of Zhejiang University SCIENCE, v. 6A, n. 6, p. 528–534, jun 2005.

        [8] XU, G. An adaptive parameter tuning of particle swarm optimization algorithm. Applied Mathematics and Computation, v. 219, n. 9, p. 4560–4569, jan 2013.
        
        [9] SCHWAAB, M.; BISCAIA, J. . E. C.; MONTEIRO, J. L.; PINTO, J. C. Nonlinear parameter estimation through particle swarm   optimization. Chemical Engineering Science, v. 63, n. 6, p. 1542–1552, mar 2008. 
        
        [10] RATNAWEERA, A.; HALGAMUGE, S. K.; WATSON, H. C. Self-Organizing Hierarchical Particle Swarm Optimizer With Time-Varying Acceleration Coefficients. IEEE Transactions on Evolutionary Computation, v. 8, n. 3, p. 240–255, jun. 2004.
                
        [11] EBERHART, R.; SHI, Y. Comparing inertia weights and constriction factors in particle swarm optimization. In: Proceedings of the 2000 Congress on Evolutionary Computation. CEC00 (Cat. No.00TH8512). IEEE, 2000. v. 1, n. 7, p. 84–88.
        """
        global vetor_posicoes, vetor_fitness, vetor_velocidades, best_fitness, gbest
        
        # Verificação se existe keyword não listada
        for key in kwargs.keys():
            if not key in ['posinit_sup', 'posinit_inf' ,'w' ,'C1' ,'C2', 'Vmax','Vreinit' , 'otimo' , 'deltaw', 'k', 'gama', 'args_model']:
                raise NameError, u'Keyword: %s not available! Available keywords: posinit_sup, posinit_inf, w, C1, C2, Vmax, Vreinit, otimo. Please verify!'%(key)

        # Valores default das kwargs (Aquelas que dependem do método escolhido, default = None):
	posinit_sup = kwargs.get('posinit_sup')  # inicializar o algoritmo em um intervalo diferente daquele definido pelo limite_superior 
	posinit_inf = kwargs.get('posinit_inf')  # inicializar o algoritmo em um intervalo diferente daquele definido pelo limite_inferior 
	w           = kwargs.get('w')            # Peso de inércia (definido None, pois seu valor default depende do método) (usado no PSO)
	C1          = kwargs.get('C1')           # Fator de aceleração do coeficiente individual (definido None, pois seu valor default depende do método)
	C2          = kwargs.get('C2')	         # Fator de aceleração do coeficiente social (definido None, pois seu valor default depende do método)
	Vmax        = kwargs.get('Vmax')         # Velocidade máxima do algoritmo
	Vreinit     = kwargs.get('Vreinit')	 # Velocidade de reiniciação (usada do HPSO): Vreinit must be a list. Structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]
	otimo 	 = kwargs.get('otimo')	 # Ponto focal para o algoritmo (usado quando o método é definido como 'Regiao')

	# Valores default das kwargs (Aquelas que não dependem do método escolhido):
	deltaw = 0.1        # Parâmetro de seleção para o algoritmo PSO-TVIW-Adaptative-VI | Default 0.1, conforme recomendação de [8]
	k  = 1.0	    # Utilizado no método CFM | Defaut 1, conforme recomendação de    [4] (Para o método CFM)
	gama = 0.5	    # Utilizado no método CFM | Default 0.5, conforme recomendação de [7] para funções com mínimos locais
	args_model=[]	    # Argumentos extras a serem enviados para o modelo

       
	# Sobrecrever os valores default, caso entrado pelo usuário
	
	if 'deltaw' in kwargs.keys():
	    deltaw = kwargs['deltaw']

	if 'k' in kwargs.keys():
	    k = kwargs['k']

	if 'gama' in kwargs.keys():
	    gama = kwargs['gama']
	
	if 'args_model' in kwargs.keys():
	    args_model = kwargs['args_model']

        # Inicialização dos métodos - Escrever métodos default, caso não definido pelo usuário
        self.metodo     = Metodo(metodo)
        self.metodo.Inicializacao()
    
        self.args_model = args_model

        # Atribuição a variáveis self, das outras variáveis
        self.gama           = gama    
        self.k              = k      
        self.deltaw         = deltaw  
        
        self.itmax          = int(itmax+1) # A iteração 0 é a inicialização
        self.Num_particulas = int(Num_particulas)
        
        self.limite_inferior = limite_inferior # Lista
        self.limite_superior = limite_superior # Lista
        self.posinit_sup     = posinit_sup     # Lista
        self.posinit_inf     = posinit_inf
        
        self.foco            = otimo

	self.default(w,C1,C2)  
	self.inicializacao(Vmax,Vreinit)

    def default(self,w,C1,C2):
        '''
        Subrotina para definição dos valores default para o PESO DE INÉRCIA E FATOR DE ACELERAÇÃO
        
        ===============        
        Peso de inércia
        ===============
        
        * Se o método para cálculo do peso de inércia é ``Constante``, sem CFM, ``w = [0.9]``, com CFM, ``w = [0.729]``. 
        * Se o método para cálculo do peso de inércia é ``TVIW-linear``, ``w = [0.9,0.4]``
        * Se o método para cálculo do peso de inércia é  ``TVIW-random``, ``w = None``
        * Se o método para cálculo do peso de inércia é  ``TVIW-Adaptative-VI``, ``w = [0.9,0.3]``
        
        ================================
        Fator de aceleração individual
        ================================
        
        * Se o método para cálculo dos fatores de aceleração é ``Constante``, sem CFM, ``C1 = [2.0]``, com CFM, ``C1 = [2.025]``. 
        * Se o método para cálculo dos fatores de aceleração é ``TVAC``, ``C1 = [2.5,0.5]`` . 
        * Se o método para cálculo do peso de inércia é ``TVIW-Adaptive-VI``, ``C1 = [1.49]`` . 

        ================================
        Fator de aceleração social
        ================================
        
        * Se o método para cálculo dos fatores de aceleração é ``Constante``, sem CFM, ``C1 = [2.0]``, com CFM, ``C1 = [2.025]``. 
        * Se o método para cálculo dos fatores de aceleração é ``TVAC``, ``C1 = [0.5,2.5]`` . 
        * Se o método para cálculo do peso de inércia é ``TVIW-Adaptive-VI``, ``C1 = [1.49]`` . 
        
        **Observe que NEM todas as combinações de métodos possuem valores *default* de parâmetros. Use com cautela**
        '''
	# Definição de valores default para w, C1, C2, caso não definidos pelo usuário:

	if self.metodo.inercia == 'Constante':        
	    if w == None:
		if self.metodo.CFM == False:
		    self.wi = 0.9 # Conforme valor recomendado por [4]
		else:
		    self.wi = 0.729
	    elif size(w) == 1:
		self.wi = w[0]
	    else:
		raise NameError, u'w deve ser uma lista de 1 valor, visto que o método para w é constante'

	elif self.metodo.inercia == 'TVIW-linear':
	    if w == None:
		self.wi = 0.9 # Conforme valor recomendado por [4]
		self.wf = 0.4 # Conforme valor recomendado por [4]
	    elif size(w) == 2:
		self.wi = w[0]
		self.wf = w[1]
	    else:
		raise NameError, u'w deve ser uma lista de 2 valores, visto que o método para w é TVIW-linear'

	elif self.metodo.inercia == 'TVIW-random':
	    if w == None:
		self.wi = None # Valor não será necessário. W é aleatório
	    else:
		raise NameError, u'w deve ser None, pois varia aleatoriamente'

	elif self.metodo.inercia == 'TVIW-Adaptative-VI':
	    if w == None:
		self.wi = 0.9 # Conforme valor recomendado por [8]
		self.wf = 0.3 # Conforme valor recomendado por [8]
	    elif size(w) == 2:
		self.wi = w[0]
		self.wf = w[1]
	    else:
		raise NameError, u'w deve ser uma lista de 2 valores, visto que o método para w é TVIW-Adaptative-VI'

	elif self.metodo.inercia == None:
	    self.wi = None

	if C1 == None:
	    if (self.metodo.algoritmo == 'PSO') or (self.metodo.algoritmo == 'HPSO'):
		if self.metodo.aceleracao == 'TVAC':
		    self.C1i = 2.5 # Default 2.5, conforme recomendado por [10]
		    self.C1f = 0.5 # Default 0.5, conforme recomendado por [10]
		elif self.metodo.inercia == 'TVIW-Adaptative-VI':
		    self.C1i = 1.49
		else:
		    if self.metodo.CFM == False:
			self.C1i = 2.0      # Default 2, conforme recomendado por [4]
		    else:
			self.C1i = 2.025  # Default 2.025, conforme recomendado por [7] para função com mínimos locais
	else:
	    if self.metodo.aceleracao == 'TVAC':
		if size(C2) == 2:
		    self.C1i = C1[0]
		    self.C1f = C1[1]
		else:
		    raise NameError, u'C2 deve ser uma lista de 2 valores para usar o método TVAC'
	    else:
		if size(C1) == 1:
		    self.C1i = C1[0]
		else:
		    raise NameError, u'C1 deve ser uma lista de 1 valor para usar o método constante'

	if C2 == None:
	    if (self.metodo.algoritmo == 'PSO') or (self.metodo.algoritmo == 'HPSO'):
		if self.metodo.aceleracao == 'TVAC':
		    self.C2i = 0.5 # Default 0.5, conforme recomendado por [10]
		    self.C2f = 2.5 # Default 2.5, conforme recomendado por [10]
		elif self.metodo.inercia == 'TVIW-Adaptative-VI':
		    self.C2i = 1.49
		else:
		    if self.metodo.CFM == False:
			self.C2i = 2.0    # Default 2, conforme recomendado por [4]
		    else:
			self.C2i = 2.025  # Default 2.025, conforme recomendado por [7] para função com mínimos locais
	else:
	    if self.metodo.aceleracao == 'TVAC':
		if size(C2) == 2:
		    self.C2i = C2[0]
		    self.C2f = C2[1]
		else:
		    raise NameError, u'C2 deve ser uma lista de 2 valores para usar o método TVAC'
	    else:
		if size(C2) == 1:
		    self.C2i = C2[0]
		else:
		    raise NameError, u'C2 deve ser uma lista de 1 valor para usar o método constante'

    def inicializacao(self,Vmax,Vreinit):
	'''
      Subrotina para inicialização das VELOCIDADES MÁXIMAS e de REINICIALIZAÇÃO, incluindo seus valores *default*. Validação dos limites de busca.
      
      ===================
      Velocidades máximas
      ===================
      
      * A velocidade máxima é mantida no range do problema
      * Caso o CFM seja utilizado, este é ``gama`` vezes o range do problema.
      
      =============================
      Velocidade de reinicialização
      =============================
      
      *  Se o método para ``Vreinit`` for ``Constante``, a velocidade de reinicialização é inicializada em Vmax, com ``final_percentage`` igual a 1
      *  Se o método para ``Vreinit`` for ``TVVr-linear``, a velocidade de reinicialização é inicializada em Vmax, com ``final_percentage`` igual a 0.1
      '''        
        
        
        
	# Validação dos limite inferior e superior
	if size(self.limite_inferior) != size(self.limite_superior):
	    raise NameError, u'Os limites (de restrição em caixa) devem ter a mesma dimensão'

	self.Num_parametros = size(self.limite_superior) # O número de parâmetros é determinado pela dimensão do limite superior e inferior

	if self.posinit_sup != None:
	    if size(self.posinit_sup) != self.Num_parametros:
		raise NameError, u'O limite de inicialização posinit_sup deve ter a mesma dimensão dos limites de restrição em caixa'

	if self.posinit_inf != None:
	    if size(self.posinit_inf) != self.Num_parametros:
		raise NameError, u'O limite de inicialização posinit_inf deve ter a mesma dimensão dos limites de restrição em caixa'

	# Forçando os limites a serem floats
	self.limite_inferior = [float(lim) for lim in self.limite_inferior]
	self.limite_superior = [float(lim) for lim in self.limite_superior]

	# tete para avaliar se o limite superior é menor do que  o inferior.
	testelimite = [self.limite_inferior[i]>=self.limite_superior[i] for i in xrange(self.Num_parametros)]
     
	if True in testelimite:
         raise ValueError, u'o limite inferior deve ser menor do que o limite superior para todas as dimensões'

	# Cálculo do módulo da velocidade máxima e validação
	if Vmax == None:
	    self.Vmax = [0]*self.Num_parametros
	    for i in xrange(self.Num_parametros):
		if (self.metodo.algoritmo == 'PSO'):
		    if self.metodo.CFM == False:
			self.Vmax[i] = max((abs(self.limite_inferior[i]),abs(self.limite_superior[i]))) # Conforme [4], Vmax é mantido no limite dinâmico do problema
		    else:
			self.Vmax[i] = self.gama*max((abs(self.limite_inferior[i]),abs(self.limite_superior[i]))) # Conforme [7], Vmax é o limite dinâmico do problema vezes gama
		elif self.metodo.algoritmo == 'HPSO':
			self.Vmax[i] = max((abs(self.limite_inferior[i]),abs(self.limite_superior[i]))) # Conforme [10], Vmax é mantido no range dinâmico do problema
	else:
	    if size(Vmax) == self.Num_parametros:
		self.Vmax = Vmax
	    else:
		raise NameError, u'Vmax deve ter a mesma dimensão do limite_inferior e limite_superior'

	# Cálculo da velocidade de reinicialização para o HPSO
	if (self.metodo.algoritmo == 'HPSO'):
	    if (Vreinit != None) and not isinstance(Vreinit,list):
		raise NameError, u'Vreinit must be a list. Structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]'
	    if (Vreinit != None) and not isinstance(Vreinit[0],list):
		raise NameError, u'The first valur of Vreinit list must be another list. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]'
	    if self.metodo.Vreinit == 'Constante':
	        if Vreinit ==  None:
		    self.Vreinit_init = [self.Vmax,1.0] # Conforme [10], a velocidade de reinicialização é mantida em Vmax
	        else:
		    if size(Vreinit[0]) == self.Num_parametros:
		        self.Vreinit_init = Vreinit
	            else:
		        raise NameError, u'The first list of Vreinit must contain a reinitialization velocity for each paramater'
		    
		    if (Vreinit[1]) != 1.0:
			raise NameError, 'The porcentage must be 1 for Vreinit, because the method chosen is Constant'
	    
	    if self.metodo.Vreinit == u'TVVr-linear':
	        if Vreinit ==  None:
		    self.Vreinit_init = [self.Vmax,0.1] # Conforme [10], a velocidade de reinicialização inicial é Vmax e decresce linearmente até 10% deste valor
	        else:
		    if size(Vreinit[0]) == self.Num_parametros:
		        self.Vreinit_init = Vreinit
	            else:
		        raise NameError, u'The first list of Vreinit must contain a reinitialization velocity for each paramater'

	else:
	    self.Vreinit = None # Não é utilizada pelos demais métodos

	# Cálculo da velocidade de inicio -> TVIW-Adaptative-VI:
	self.Vstart = abs((max(self.limite_superior) - min(self.limite_inferior))/2.0)
	self.Tend   = 0.95*self.itmax

    def Busca(self,FO,printit=False):
	'''
	Método para realizar a etapa de busca na função objetivo
	
	===================
	Entrada obrigatória
	===================
	
	* ``FO`` : função objetivo no formato de Thread com atributo de resposta ``result`` (sendo um float). Exemplo: ::
	
	    from threading import Thread
            
            class FO(Thread):
                result = 0
                def __init__(self,param,args):
                    Thread.__init__(self)
                    self.x = param
    
                def run(self):
                    
                    self.result =  self.x**2
	
	================
	Entrada opcional
	================
	
	* ``printit`` (bool: True ou False): se True o número das iterações, ao decorrer da busca, são apresentadas em tela.
	'''
        global Controle_FO, Controle_Particula, Controle_Iteracao, Controle_variaveis, total_particulas_atendidas, Controle_Total_Threads
        global vetor_posicoes, vetor_fitness, vetor_velocidades, best_fitness, gbest # Apenas para o método de gbest Enxame
        
	# FO: função objetivo -> tem DE ser uma classe herdando os atributos da classe Thread, os resultados tem
	#		         DE ser um número REAL, sendo disponibilizados em método result (FO.result).
	
 
        if (self.itmax-1)==1 and (self.metodo.inercia == 'TVIW-linear' or self.metodo.aceleracao == 'TVAC'):
            raise ValueError, u'Para os métodos de peso de inércia e aceleraçao com variaçao linear (TVIW-linear e TVAC), o número de iterações deve ser, no mínimo, 2'
 
        # Controles de Threads | Semáforo
        max_Threads_permitido = 50
        
        if self.Num_particulas <= max_Threads_permitido:
            num_Threads = self.Num_particulas
            
        else:
            num_Threads = max_Threads_permitido
        
        Controle_Particula = BoundedSemaphore(value=num_Threads) # Semáforo -> Controlar o número máximo de Threads a serem utilizadas pelo Programa (evitar número excessivo) ao mesmo tempo para avaliação das partículas no def Busca
        Controle_FO        = BoundedSemaphore(value=num_Threads) # Semáforo -> Controlar o número máximo de Threads a serem utilizadas pelo Programa (evitar número excessivo) ao mesmo tempo para avaliação da função objetivo no def __init__

        # Inicializações de listas: posiçoes, velocidades e fitness
        
        vetor_posicoes    = []
        vetor_velocidades = []
        vetor_fitness     = []
        
        for i in xrange(self.itmax):
            #Posição
            aux1 = [0]*self.Num_particulas
            vetor_posicoes.append(aux1)
            # Velocidade
            aux2 = [0]*self.Num_particulas
            vetor_velocidades.append(aux2)
            #Fitness
            aux3 = [1e100]*self.Num_particulas
            vetor_fitness.append(aux3)
    
        #Inicialização das posicoes, velocidades e fitness
        
        threadarray = []
        ID_FO       = 0
        ID_particle = 0
        
        while (ID_particle < self.Num_particulas) or (ID_FO < self.Num_particulas) :
            
            # Fitness
            if ID_particle > 0:
                if not(threadarray[ID_FO].isAlive()):
                    if size(threadarray[ID_FO].result)>1:
                        raise NameError, u'A função objetivo possui mais de uma dimensão. Verificá-la'
                    else:
                        vetor_fitness[0][ID_FO] = float(copy(threadarray[ID_FO].result))
                        ID_FO +=1
                        Controle_FO.release()
            
            if ID_particle < self.Num_particulas:
                
                # Posição e velocidade
                pos = []
                vel = []
                for p in xrange(self.Num_parametros):
                    random.seed(ID_particle+p*2+245)
                    if (self.posinit_sup == None) and (self.posinit_inf == None):
                        pos.append(random.uniform(self.limite_inferior[p],self.limite_superior[p]))
                        vel.append(random.uniform(self.limite_inferior[p],self.limite_superior[p]))
                    elif (self.posinit_sup != None) and (self.posinit_inf == None):
                        pos.append(random.uniform(self.limite_inferior[p],self.posinit_sup[p]))
                        vel.append(random.uniform(self.limite_inferior[p],self.posinit_sup[p]))
                    elif (self.posinit_sup == None) and (self.posinit_inf != None):
                        pos.append(random.uniform(self.posinit_inf[p],self.limite_superior[p]))
                        vel.append(random.uniform(self.posinit_inf[p],self.limite_superior[p]))
                    elif (self.posinit_sup != None) and (self.posinit_inf != None):
                        pos.append(random.uniform(self.posinit_inf[p],self.posinit_sup[p]))
                        vel.append(random.uniform(self.posinit_inf[p],self.posinit_sup[p]))
            
		auxp = copy(pos)
		auxv = copy(vel)
                vetor_posicoes[0][ID_particle]    = auxp.tolist()
                vetor_velocidades[0][ID_particle] = auxv.tolist()

                # Fitness
                Controle_FO.acquire()
                fitness = FO(vetor_posicoes[0][ID_particle],self.args_model)
                fitness.start()
                threadarray.append(fitness)
                ID_particle+=1
        
	# validação de gbest foi definido para o método de busca Regiao
        if self.metodo.busca == 'Regiao':
            if self.foco == None:
                raise ValueError, u'É necessário informar o valor de um ponto ótimo da função'
            vetor_posicoes[0][0] = self.foco
        
        
        # Inicializando melhores valores globais do enxame
        self.best_ID_particle = argmin(vetor_fitness[0])
        best_fitness     = vetor_fitness[0][self.best_ID_particle]
        if self.metodo.busca == 'Otimo':
            gbest            = vetor_posicoes[0][self.best_ID_particle]
        elif self.metodo.busca == 'Regiao':
            gbest            = self.foco
            
        # Inicialização dos vetores médios e de W, para plotagem de gráficos (def Gráficos):
        self.media_fitness             = [0]*self.itmax
        self.media_velocidade          = [0]*self.itmax
        self.velocidade_ideal          = [0]*self.itmax
        self.desvio_fitness            = [0]*self.itmax
        self.historico_w               = [0]*self.itmax        
        self.historico_best_fitness    = [0]*self.itmax
        
        self.media_fitness[0]          = mean(vetor_fitness[0])
        self.desvio_fitness[0]         = std(vetor_fitness[0],ddof=1)
        self.historico_best_fitness[0] = best_fitness

        self.historico_w[0]            = self.wi
        
        aux1 = []
        for j in xrange(self.Num_particulas):
            aux2 = [abs(vetor_velocidades[0][j][k]) for k in xrange(self.Num_parametros)]
            aux1.append(aux2)
        
        self.media_velocidade[0]       = mean(aux1)
        self.velocidade_ideal[0]       = self.Vstart
	
	# Definição de Locks (para controle das interações e regiões críticas)
        Controle_Iteracao      = Lock() # Aplicável para evitar que a iteração acabe prematuramente, devido ao não término da execução de Threads
        Controle_variaveis     = Lock() # Aplicável para controlar a região crítica envolvendo as variáveis característica das partículas (posicão, velocidade, fitness e gbest)
        Controle_Total_Threads = Lock() # Aplicável para controlar a região crítica envolvendo a variável total_particulas_atendidas

        # Partículas
        vetor_Num_particulas = range(self.Num_particulas) # Vetor_ID_PARTICULA

        # Iterações
        for it in xrange(1,self.itmax): # a iteração 0 é a inicialização
            
            total_particulas_atendidas = 0 # Contagem de partículas que finalizaram a execução
            
            # Aquisição de Controle_Iteração, para iniciar a iteração
            Controle_Iteracao.acquire()
	    if printit == True:
		sys.stdout.write('ITERACAO: '+str(it)+'\n')
		sys.stdout.flush()
            
	    # Atualização de w, C1, C2 e Vreinit
            if self.metodo.inercia == 'Constante':
                self.w = self.wi
            elif self.metodo.inercia == 'TVIW-linear':
                self.w = self.wi + (float(it)-1)/(1-float(self.itmax-1))*(self.wi-self.wf)
            elif self.metodo.inercia == 'TVIW-random':
                self.w = 0.5 + random.uniform(0.0,1.0)/2
            elif self.metodo.inercia == 'TVIW-Adaptative-VI':

                self.velocidade_ideal[it] = self.Vstart* (1+cos(it*pi/self.Tend))/2
                
                if self.media_velocidade[it-1] >= self.velocidade_ideal[it]:
                    self.w = max([self.historico_w[it-1]-self.deltaw,self.wf])
                else:
                    self.w = min([self.historico_w[it-1]+self.deltaw,self.wi])
            elif self.metodo.inercia == None:
                self.w = None    

            if self.metodo.aceleracao == 'Constante':
                self.C1 = self.C1i
                self.C2 = self.C2i
            elif self.metodo.aceleracao == 'TVAC':
                self.C1 = self.C1i + (float(it)-1)/(1-float(self.itmax-1))*(self.C1i-self.C1f)
                self.C2 = self.C2i + (float(it)-1)/(1-float(self.itmax-1))*(self.C2i-self.C2f)
  
            if self.metodo.algoritmo == 'HPSO':
		self.Vreinit = []
		for i in xrange(self.Num_parametros):
		    self.Vreinit.append(self.Vreinit_init[0][i] + (float(it)-1)/(1-float(self.itmax-1))*(self.Vreinit_init[0][i]-self.Vreinit_init[1]*self.Vreinit_init[0][i]))    

            # Create new Threads (Partículas)
            for ID_particle in vetor_Num_particulas:
                Controle_Particula.acquire() # Aquisião do Semáforo -> limitar a quantidade de particulas sendo executadas ao mesmo tempo
                particula = Particula(FO,self.w,self.C1,self.C2,self.k,self.Vmax,self.Vreinit,it,self.Num_parametros,self.Num_particulas,self.metodo,self.limite_superior,self.limite_inferior,ID_particle,self.args_model)
                particula.start()
            
            # A execução aguarda a liberação (release) de Controle_Iteração
            Controle_Iteracao.acquire()
            
            if self.metodo.gbest == 'Enxame':
                if min(vetor_fitness[it])<best_fitness:
                    best_loc_part = argmin(vetor_fitness[it])
                    best_fitness  = vetor_fitness[it][best_loc_part]
                    gbest         = vetor_posicoes[it][best_loc_part]
            
            # Cálculo de parâmetros de avaliação, médias das iterações.
            self.media_fitness[it]          = mean(vetor_fitness[it])
            self.desvio_fitness[it]         = std(vetor_fitness[it],ddof=1)
            self.historico_w[it]            = self.w
            self.historico_best_fitness[it] = best_fitness
            
            aux1 = []
            for j in xrange(self.Num_particulas):
                aux2 = [abs(vetor_velocidades[it][j][k]) for k in xrange(self.Num_parametros)]
                aux1.append(aux2)
            
            self.media_velocidade[it]       = mean(aux1)
 
            # Liberação da proxima iteração
            Controle_Iteracao.release()
    
        # Armazenamento das informações em variáveis da classe (disponíveis para o Programa Principal)
        self.historico_fitness  = vetor_fitness    
        self.historico_posicoes = vetor_posicoes
        self.gbest              = gbest
        self.best_fitness       = best_fitness

	for it in xrange(self.itmax):
	    for ID_particula in xrange(self.Num_particulas):
		if isnan(self.historico_fitness[it][ID_particula]):
		    raise NameError, u'Existe NaN como valor de função objetivo. Verificar.'
    
    def Result_txt(self,base_path=None):
        '''
        Método para gerar os arquivos de texto contendo os resultados
        
        =======
        Entrada
        =======
        
        * ``base_path`` (string,opcional): caminho completo onde os arquivos serão salvos.

        =========
        Exemplo 1        
        =========
        
        Considere o **exemplo 1** da documentação do PSO. Os resultados serão salvos dentro da pasta "Resultados" que está, por sua vez, dentro de uma outra pasta "Exemplo" ::
     
             >>> from funcaoobjetivo import FO
             >>> import os
             >>> sup  = [10.]  # limite superior de busca para x
             >>> inf  = [-10.] # limite inferir de busca para x
             >>> Otimizacao = PSO(sup,inf) # Criação da classe PSO
             >>> Otimizacao.Busca(FO)      # Comando para iniciar a busca
             >>> base_path = os.getcwd() + '/Exemplo'
             >>> Otimizacao.Result_txt(base_path+'/Resultado/')   # Salvar os principais resultados em arquivos de texto
             >>> Otimizacao.Graficos(base_path+'/Graficos/')     # Criação de gráficos com indicadores de desempenho do algoritmo de PSO            
             
             
        ================
        Arquivos gerados
        ================
        
        * **Best_fitness.txt** :contém o valor da função objetivo avaliada no ponto ótimo
        * **gbest.txt** : contém o valor do ponto ótimo encontrado pelo algoritmo de PSO
        * **historico_posicoes.txt** : contém o histórico das posições avaliados para cada particula a cada iteração.
        * **historico_fitness.txt**: contém o histórico de avaliação da função objetivo para cada particula a cada iteração.
        * **Resumo.txt** : contém um resumo da otimização realizada
        '''        
        
        if base_path == None:
            base_path = os.getcwd()+"/PSO/Result_txt/"
        Validacao_Diretorio(base_path)
    
        savetxt(base_path+'Best_fitness.txt'      ,  matrix(self.best_fitness), fmt= '%.18f')
	outfile = open(base_path+'historico_fitness.txt','w')
	for elemento in self.historico_fitness:
	    outfile.write(str(elemento))
	    outfile.write('\n')
	outfile.close()
	
	outfile = open(base_path+'historico_posicoes.txt','w')
	for elemento in self.historico_posicoes:
	    outfile.write(str(elemento))
	    outfile.write('\n')
	outfile.close()

        savetxt(base_path+'gbest.txt'             ,  matrix(self.gbest)       , fmt= '%.18f')
    
	# Resumo da estimação
	outfile = codecs.open(base_path+'Resumo.txt','w','utf-8')
	outfile.write(u'RESUMO DO PSO\n')
	outfile.write('--------------------------------'+'\n')
	outfile.write(u'MÉTODO'+'\n')
	outfile.write(u'Algoritmo: %s | Inercia: %s | Aceleracão: %s | Vreinit: %s'%(self.metodo.algoritmo,\
	self.metodo.inercia,self.metodo.aceleracao,self.metodo.Vreinit)+'\n')
	outfile.write('--------------------------------'+'\n')
	outfile.write(u'Seleção dos parâmetros, partículas e iterações'+'\n')
	outfile.write(u'INÍCIO - w: %s | c1: %s | c2: %s'%(self.wi,self.C1i,self.C2i)+'\n')
	if self.metodo.inercia == 'Constante':
	    wf = self.wi
	elif self.metodo.algoritmo != 'HPSO' and self.metodo.inercia != 'TVIW-random':
          wf = self.wf
	elif self.metodo.algoritmo == 'HPSO' or self.metodo.inercia == 'TVIW-random':
	    wf = None
	if self.metodo.aceleracao == 'Constante':
	    C1f = self.C1i
	    C2f = self.C2i
	else:
	    C1f = self.C1f
	    C2f = self.C2f
	outfile.write(u'FIM    - w: %s | c1: %s | c2: %s'%(wf,C1f,C2f)+'\n')
	outfile.write(u'Vmax: %s'%self.Vmax+'\n')
	outfile.write(u'Vreinit: %s'%self.Vreinit+'\n')
	outfile.write(u'deltaw: %s'%self.deltaw+'\n')
	outfile.write(u'Número de partículas: %s'%(self.Num_particulas,)+'\n')
	outfile.write(u'Iterações: %s (1 de inicialização, %s de Busca)'%(self.itmax,self.itmax-1)+'\n')
	outfile.write('--------------------------------'+'\n')
	outfile.write(u'RESULTADOS'+'\n')
	outfile.write(u'ótimo: %s | fitness: %s'%(self.gbest,self.best_fitness))
	outfile.close()
    
    def Graficos(self,base_path=None,**kwargs):
	'''
	Método para criação dos gráficos de indicadores do PSO, incluindo a função objetivo.
 
     =========
     Entradas
     =========
     
	* ``base_path`` (string, opcional): caminho completo onde os gráficos serão criados.
 
     =========
     Exemplo 1
     =========
     Considere o **exemplo 1** da documentação do PSO. Os gráficos serão salvos dentro da pasta "Graficos" que está, por sua vez, dentro de uma outra pasta "Exemplo" ::
     
         >>> from funcaoobjetivo import FO
         >>> import os
         >>> sup  = [10.]  # limite superior de busca para x
         >>> inf  = [-10.] # limite inferir de busca para x
         >>> Otimizacao = PSO(sup,inf) # Criação da classe PSO
         >>> Otimizacao.Busca(FO)      # Comando para iniciar a busca
         >>> base_path = os.getcwd() + '/Exemplo'
         >>> Otimizacao.Result_txt(base_path+'/Resultado/')   # Salvar os principais resultados em arquivos de texto
         >>> Otimizacao.Graficos(base_path+'/Graficos/')     # Criação de gráficos com indicadores de desempenho do algoritmo de PSO     
     
     ========
     Keywargs
     ========
     
     * ``legenda_posicao`` (número intreiro): define a posição da legenda nos gráficos. Valores: 1 (superior direita), 2 (superior esquerda), 3 (inferior esquerda), 4 (inferior direita), 5 (direita), 6 (centro esquerda), 7 (centro direita), 8 (centro inferior),  9 (centro superior), 10 (centro). O valor default é 1.
     * ``azim`` (lista): define a rotação horizontal para os gráficos da função objetivo. Se tiver apenas um elemento, será utilizado o valor para todos os gráficos.  O valor default é ``[45]``
     * ``elev`` (lista): define a rotação vertical para os gráfico da função objetivo. O valor default é ``[45]``.
     * ``Nome_param`` (lista): define o nome dos parâmetros, na ordem que incluidos nos limites. Valor default ``[x1,x2,...,xn]``. Pode ser utilizado os códigos em LATEX, só utilizar: r'$codigo$'
     * ``Unid_param`` (lista): define o nome dos parâmetros, na ordem que incluidos nos limites. Valor default ``[adim,adim,...,adim]``. Pode ser utilizado os códigos em LATEX, só utilizar: r'$codigo$
     * ``FO2a2`` (bool): define se o gráfico da função objetivo é realizado também para os pares de parâmetros. Valor default: False (O gráfico da função objetivo é feito apenas para cada parâmetro separadamente). Qualquer entrada diferente é assumida como True.
      '''
	# kwargs: legenda_posicao, azim, elev, Nome_param, Unid_param
	# Atribuição de valores default
	legenda_posicao = 1
	azim = [45] # Define a rotação horizontal no gráfico da função objetivo
	elev = [45] # Define a rotação vertical no gráfico da função objetivo
	Nome_param = []
	Unid_param = []
	FO2a2      = False
	# Posições de Legenda:"
        #   Superior Direita  = 1
        #   Superior Esquerda = 2
        #   Inferior Esquerda = 3
        #   Inferior Direita  = 4
        #   Direita           = 5
        #   Centro Esquerda   = 6
        #   Centro Direita    = 7
        #   Centro inferior   = 8
        #   Centro Superior   = 9
        #   Centro            = 10
	
	# Sobreescrevendo valores default
	if 'legenda_posicao' in kwargs.keys():
	    legenda_posicao = kwargs['legenda_posicao']
	
	if 'azim' in kwargs.keys():
	    azim = kwargs['azim']
	
	if 'elev' in kwargs.keys():
	    elev = kwargs['elev']
	
	if 'Nome_param' in kwargs.keys():
	    Nome_param = kwargs['Nome_param']
	    if Nome_param != None:
		if len(Nome_param) != self.Num_parametros and len(Nome_param) != 0 :
		    raise NameError, u'Nome_param deve conter o nome para todos os parâmetros'
	    
	if 'Unid_param' in kwargs.keys():
	    Unid_param = kwargs['Unid_param']
	    if Unid_param != None:
		if len(Unid_param) != self.Num_parametros and len(Unid_param) != 0:
		    raise NameError, u'Unid_param deve conter as unidades para todos os parâmetros'

	if 'FO2a2' in kwargs.keys():
	    if kwargs['FO2a2'] == False:
		FO2a2 = False
	    else:
		FO2a2 = True
		
	if isinstance(azim, int) or isinstance(azim, float) or isinstance(elev, int) or isinstance(elev,float):
	    raise NameError, u'keywords azim e elev devem ser listas'
	
        # Verificação da existência de diretório do diretório
        if base_path == None:
            base_path = os.getcwd()+"/PSO/Graficos/"
        Validacao_Diretorio(base_path)
        
	# Gráficos de Desempenho do algoritmo de PSO-----------------------------------------------
        # Início
        iteracoes = range(self.itmax)
         
	# Uso do formato científico nos gráficos:
	formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2,2))
        # Desenvolvimento das iterações - Fitness e desvio do fitness
        fig = figure()
        ax = fig.add_subplot(2,1,1)
        plot(iteracoes[0:self.itmax],self.media_fitness[0:self.itmax],'b-',label = u'Média')
        plot(iteracoes[0:self.itmax],self.historico_best_fitness[0:self.itmax],'r-',label = u'Best Fitness')
        ax.yaxis.grid(color='gray', linestyle='dashed')                        
        ax.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((0,self.itmax))               
        ylabel(u"Média de "+r"$\Phi$", fontsize = 15)
        legend(loc=legenda_posicao)
        
        ax2 = fig.add_subplot(2,1,2)
        plot(iteracoes[0:self.itmax],self.desvio_fitness[0:self.itmax],'b-',label = u'Desvio-padrão')
        ax2.yaxis.grid(color='gray', linestyle='dashed')                        
        ax2.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((0,self.itmax))               
        xlabel(u"Iterações", fontsize = 15)
        ylabel(u"Desvio-padrão de "+r"$\Phi$", fontsize = 15)
        legend(loc=legenda_posicao)
        ax.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path+'Medias_Fitness_global.png')
        close()

        # Desenvolvimento das 50% primeiras iterações
        fig = figure()
        ax = fig.add_subplot(2,1,1)
        plot(iteracoes[0:self.itmax/2],self.media_fitness[0:self.itmax/2],'b-',label = u'Média')
        plot(iteracoes[0:self.itmax/2],self.historico_best_fitness[0:self.itmax/2],'r-',label = u'Best Fitness')
        ax.yaxis.grid(color='gray', linestyle='dashed')                        
        ax.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((0,self.itmax/2))
        legend(loc=legenda_posicao)
        ylabel(u"Média de " + "$\Phi$")
        #
        ax2 = fig.add_subplot(2,1,2)
        plot(iteracoes[0:self.itmax/2],self.desvio_fitness[0:self.itmax/2],'b-',label = u'Desvio-padrão')
        ax2.yaxis.grid(color='gray', linestyle='dashed')                        
        ax2.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((0,self.itmax/2))               
        xlabel(u"Iterações")
        ylabel(u"Desvio-padrão de " + r"$\Phi$")
        legend(loc=legenda_posicao)
        ax.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path+'Medias_Fitness_iteracoes_iniciais.png')
        close()

        # Desenvolvimento das 50% últimas iterações
        fig = figure()
        ax = fig.add_subplot(2,1,1)
        plot(iteracoes[self.itmax/2:self.itmax],self.media_fitness[self.itmax/2:self.itmax],'b-',label = u'Média')
        plot(iteracoes[self.itmax/2:self.itmax],self.historico_best_fitness[self.itmax/2:self.itmax],'r-',label = u'Best Fitness')
        ax.yaxis.grid(color='gray', linestyle='dashed')                        
        ax.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((self.itmax/2,self.itmax))
        legend(loc=legenda_posicao)
        ylabel(u"Média de" + r"$\Phi$")
        #
        ax2 = fig.add_subplot(2,1,2)
        plot(iteracoes[self.itmax/2:self.itmax],self.desvio_fitness[self.itmax/2:self.itmax],'b-',label = u'Desvio-padrão')
        ax2.yaxis.grid(color='gray', linestyle='dashed')                        
        ax2.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((self.itmax/2,self.itmax))               
        xlabel(u"Iterações")
        ylabel(u"Desvio-padrão de "+ r"$\Phi$")
        legend(loc=legenda_posicao)
	ax.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path+'Medias_Fitness_iteracoes_finais.png')
        close()
        
        # Desenvolvimento das Iterações - Velocidades
        fig = figure()
        ax = fig.add_subplot(1,1,1)
        plot(iteracoes[0:self.itmax],self.media_velocidade[0:self.itmax],'b-')
        if self.metodo.inercia == 'TVIW-Adaptative-VI':
            plot(iteracoes[0:self.itmax],self.velocidade_ideal[0:self.itmax],'r-')
        ax.yaxis.grid(color='gray', linestyle='dashed')                        
        ax.xaxis.grid(color='gray', linestyle='dashed')         
        xlim((0,self.itmax))
        xlabel(u"Iterações", fontsize = 15)
        ylabel(u"Média de "+r"$\nu$", fontsize = 15)
        ax.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path+'Medias_velocidade_global.png')
        close()
        
        # Desenvolvimento das Iterações (Evolução do algoritmo)
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i in xrange(self.itmax):
            xs = [i]*self.Num_particulas
            ys = range(0,self.Num_particulas)
            zs = self.historico_fitness[i]
            ax.scatter(xs, ys, zs, c='b', marker='o')
        
        ax.set_xlabel(u'Iteração', fontsize = 15)
        ax.set_ylabel(u'ID_Partícula', fontsize = 15)
        ax.set_zlabel(r'$\Phi$', fontsize = 15)
        ax.w_zaxis.set_major_formatter(formatter)
        fig.savefig(base_path+'Evolucao_algoritmo.png')
        close()
        	
        if (self.metodo.inercia == 'TVIW-linear') or (self.metodo.inercia == 'TVIW-random') or (self.metodo.inercia == 'TVIW-Adaptative-VI'):
            # Gráfico de histório do inertia weight
            fig = figure()
            ax3 = fig.add_subplot(1,1,1)
            plot(iteracoes,self.historico_w,'b-')
            ax3.yaxis.grid(color='gray', linestyle='dashed')                        
            ax3.xaxis.grid(color='gray', linestyle='dashed')         
            xlim((0,self.itmax))
            if self.metodo.inercia == 'TVIW-linear':
                ylim((self.wf,self.wi))
            xlabel(u"Iterações")
            ylabel(u"W")
	    ax3.yaxis.set_major_formatter(formatter)
            fig.savefig(base_path+'W_inertia_weight.png')
            close()

	# Transformação do histórico das posições e do fitness
	hist_posicoes = []; hist_fitness = []
	for it in xrange(self.itmax):
	    for ID_particula in xrange(self.Num_particulas):
		hist_posicoes.append(self.historico_posicoes[it][ID_particula])
		hist_fitness.append(self.historico_fitness[it][ID_particula])

	
	# Histograma das posições
	for D in xrange(self.Num_parametros):
	    aux = [hist_posicoes[it][D] for it in xrange(self.itmax+1)] # obtenção das posições para a dimensão D
	    
	    fig = figure()
	    ax = fig.add_subplot(1,1,1)
	    hist(aux,normed=True,histtype='stepfilled',color='b')
	    ylabel(u'Frequência normalizada')
	    xlabel(u'Valores dos parâmetros')
	    fig.savefig(base_path+'Histograma_parametros_%d'%(D+1)+'.png')
	    
	# Gráfico da função objetivo----------------------------------------------
	for ix in xrange(self.Num_parametros):
	    # Gráfico unidimensional da função objetivo
	    # Desenvolvimento das Iterações - Velocidades
	    X = [hist_posicoes[it][ix] for it in xrange(len(hist_posicoes))]
	    Y = hist_fitness
		
	    X_sort   = sort(X)
	    Y_sort   = [Y[i] for i in argsort(X)]    
   
	    fig = figure()
	    ax = fig.add_subplot(1,1,1)
	    plot(X_sort,Y_sort,'bo',markersize=4)
	    plot(self.gbest[ix],self.best_fitness,'ro',markersize=8)
	    ax.yaxis.grid(color='gray', linestyle='dashed')                        
	    ax.xaxis.grid(color='gray', linestyle='dashed')
	    ylabel(r"$\quad \Phi $",fontsize = 20)
	    
	    if  isinstance(Nome_param, list):
		if len(Nome_param) == 0:
		    xlabel(r"$x_{%d}$ "%(ix), fontsize = 20)
		elif isinstance(Unid_param,list):
		    if len(Unid_param) == 0:
			xlabel(Nome_param[ix], fontsize = 20)
		elif Unid_param == None:
		    xlabel(Nome_param[ix], fontsize = 20)
		else:    
		    xlabel(Nome_param[ix] + u'/' + Unid_param[ix], fontsize = 20)
	    else:
		xlabel(r"$x_{%d}$ "%(ix+1), fontsize = 20)
	
	    ax.yaxis.set_major_formatter(formatter)
	    fig.savefig(base_path+'Funcao_objetivo_projecao_'+'x%d'%(ix+1)+'.png')
	    close()

	if (self.Num_parametros != 1) and (FO2a2 == True):
	    # Gráfico tridimensioal da função objetivo
	    Combinacoes = int(factorial(self.Num_parametros)/(factorial(self.Num_parametros-2)*factorial(2)))
	    p1 = 0; p2 = 1; cont = 0; passo = 1
    
	    for pos in xrange(Combinacoes):
		if pos == (self.Num_parametros-1)+cont:
		    p1 +=1; p2 = p1+1; passo +=1
		    cont += self.Num_parametros-passo
		X = [hist_posicoes[it][p1] for it in xrange(len(hist_posicoes))]
		Y = [hist_posicoes[it][p2] for it in xrange(len(hist_posicoes))]
		Z = hist_fitness
		
		Z_sort   = sort(Z)
		Y_sort   = [Y[i] for i in argsort(Z)]
		X_sort   = [X[i] for i in argsort(Z)]    
    
		fig = figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X_sort,Y_sort,Z_sort,c=Z_sort,cmap=cm.coolwarm,zorder=1)
		arrow = Arrow3D([self.gbest[p1],self.gbest[p1]],[self.gbest[p2],self.gbest[p2]],[Z_sort[-1]/4,Z_sort[0]], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
		ax.add_artist(arrow)
		ax.yaxis.grid(color='gray', linestyle='dashed')                        
		ax.xaxis.grid(color='gray', linestyle='dashed')
		ax.set_zlabel(r"$\quad \Phi $ ",rotation = 180, fontsize = 20)
    
		if  isinstance(Nome_param, list):
		    if len(Nome_param) == 0:
			ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
			ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)
		    elif isinstance(Unid_param,list):
			if len(Unid_param) == 0:
			    ax.set_ylabel(Nome_param[p2], fontsize = 20)
			    ax.set_xlabel(Nome_param[p1], fontsize = 20)
		    elif Unid_param == None:
			ax.set_ylabel(Nome_param[p2], fontsize = 20)
			ax.set_xlabel(Nome_param[p1], fontsize = 20)
		    else:    
			ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize = 20)
			ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize = 20)
		else:
		    ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
		    ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)		
		
		if len(azim) == 1:
		    ax.azim = azim[0]
		else:
		    ax.azim = azim[pos]    
		
		if len(elev) == 1:
		    ax.elev = elev[0]
		else:
		    ax.elev = elev[pos]	    
		ax.set_zlim([Z_sort[0],Z_sort[-1]])
		ax.w_zaxis.set_major_formatter(formatter)
		fig.savefig(base_path+'Funcao_objetivo'+'_x'+str(p1+1)+'_x'+str(p2+1)+'.png')
		close()
		p2+=1


    def Movie(self,base_path=None,tipos=['evolucao','projecao','funcao'],**kwargs):
        '''
        Método para criação dos gráficos de desenvolvimento do algoritmo a cada iteração.
	Cada gráfico representa um frame
        =========
        Entradas
        =========
     
         * ``base_path`` (string, opcional): caminho completo onde os gráficos serão criados.
         * ``tipos´´ (lista,opcional): lista contendo os tipos de gráficos a serem criados: evolucao (gráfico 3d contendo a iteraçao, partícula e funcao objetivo), projecao (gráfico 2d contendo as posiçoes de cada particula), funcao (gráfico 3d contendo a posiçao das particulas e o valor da funçao objetivo)
         
        ========
        Keywargs
        ========
     
        * ``azim`` (lista): define a rotação horizontal para os gráficos da função objetivo. Se tiver apenas um elemento, será utilizado o valor para todos os gráficos.  O valor default é ``[45]``
        * ``elev`` (lista): define a rotação vertical para os gráfico da função objetivo. O valor default é ``[45]``.
        * ``Nome_param`` (lista): define o nome dos parâmetros, na ordem que incluidos nos limites. Valor default ``[x1,x2,...,xn]``. Pode ser utilizado os códigos em LATEX, só utilizar: r'$codigo$'
        * ``Unid_param`` (lista): define o nome dos parâmetros, na ordem que incluidos nos limites. Valor default ``[adim,adim,...,adim]``. Pode ser utilizado os códigos em LATEX, só utilizar: r'$codigo$
        '''
        # Atribuição de valores default
        azim = [45] # Define a rotação horizontal no gráfico da função objetivo
        elev = [45] # Define a rotação vertical no gráfico da função objetivo
        Nome_param = []
        Unid_param = []
        
        # Sobrescrevendo valores default
	
        if 'azim' in kwargs.keys():
            azim = kwargs['azim']
	
        if 'elev' in kwargs.keys():
            elev = kwargs['elev']
	
        if 'Nome_param' in kwargs.keys():
            Nome_param = kwargs['Nome_param']
            if Nome_param != None:
                if len(Nome_param) != self.Num_parametros and len(Nome_param) != 0 :
                    raise NameError, u'Nome_param deve conter o nome para todos os parâmetros'
	    
        if 'Unid_param' in kwargs.keys():
            Unid_param = kwargs['Unid_param']
            if Unid_param != None:
                if len(Unid_param) != self.Num_parametros and len(Unid_param) != 0:
                    raise NameError, u'Unid_param deve conter as unidades para todos os parâmetros'
                    
        if isinstance(azim, int) or isinstance(azim, float) or isinstance(elev, int) or isinstance(elev,float):
            raise NameError, u'keywords azim e elev devem ser listas'

        len_itmax = len(str(self.itmax))
        # Verificação da existência de diretório do diretório
        if base_path == None:
            base_path = os.getcwd()+"/PSO/"
        dir1 = base_path+"/Evolucao/"
        Validacao_Diretorio(dir1)
 
        # Formatação dos gráficos
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2,2))
        
        if 'evolucao' in tipos:
            # Desenvolvimento das iterações:
            for i in xrange(self.itmax):
                
                fig = figure()
                ax = fig.add_subplot(111, projection='3d')
                xs = [i]*self.Num_particulas
                ys = range(0,self.Num_particulas)
                zs = self.historico_fitness[i]
                ax.scatter(xs, ys, zs, c='b', marker='o')
                ax.set_xlabel(u'Iteração')
                ax.set_ylabel(u'ID_Partícula')
                ax.set_zlabel(u'Fitness / u.m')
                xlim((0.0,self.itmax))
                len_it = len(str(i))
                numeracao = '0'*(len_itmax - len_it) + str(i)
                fig.savefig(dir1+'Evolucao_algoritmo_'+numeracao+'.png')
                close()
        
        if self.Num_parametros>=2 and (('projecao' in tipos) or ('funcao' in tipos)):
            
            Combinacoes = int(factorial(self.Num_parametros)/(factorial(self.Num_parametros-2)*factorial(2)))
            
            for it in xrange(self.itmax):
                p1 = 0; p2 = 1; cont = 0; passo = 1
                for pos in xrange(Combinacoes):
                    dir2 = base_path+"/Projecao_Combinacoes_"+str(pos)+"/"
                    Validacao_Diretorio(dir2)
                    dir3 = base_path+"/Funcao_Combinacoes_"+str(pos)+"/"
                    Validacao_Diretorio(dir3)
                    if pos == (self.Num_parametros-1)+cont:
                        p1 +=1
                        p2 = p1+1
                        passo +=1
                        cont += self.Num_parametros-passo
                    
                    len_it = len(str(it))
                    numeracao = '0'*(len_itmax - len_it) + str(it)

                    if 'projecao' in tipos:
                        fig = figure()
                        ax = fig.add_subplot(1,1,1)
                        
                        for ID in xrange(self.Num_particulas):
                            plot(self.historico_posicoes[it][ID][p1],self.historico_posicoes[it][ID][p2],'bo',linewidth=2.0)
        
                        plot(self.gbest[p1],self.gbest[p2],'r*',markersize=10.0)
                        ax.yaxis.grid(color='gray', linestyle='dashed')                        
                        ax.xaxis.grid(color='gray', linestyle='dashed')        
                        
                        if  isinstance(Nome_param, list):
                		    if len(Nome_param) == 0:
                			ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
                			ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)
                		    elif isinstance(Unid_param,list):
                			if len(Unid_param) == 0:
                			    ax.set_ylabel(Nome_param[p2], fontsize = 20)
                			    ax.set_xlabel(Nome_param[p1], fontsize = 20)
                		    elif Unid_param == None:
                			ax.set_ylabel(Nome_param[p2], fontsize = 20)
                			ax.set_xlabel(Nome_param[p1], fontsize = 20)
                		    else:    
                			ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize = 20)
                			ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize = 20)
                        else:
                		    ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
                		    ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)		                        
                      
                        xlim((self.limite_inferior[p1],self.limite_superior[p1]))
                        ylim((self.limite_inferior[p2],self.limite_superior[p2]))

                        fig.savefig(dir2+'Parametros'+'_p'+str(p1+1)+'_p'+str(p2+1)+'_it'+numeracao+'.png')
                        close()

                    if 'funcao' in tipos:
            		hist_posicoes = []; hist_fitness = []
            		for ID_particula in xrange(self.Num_particulas):
                    		hist_posicoes.append(self.historico_posicoes[it][ID_particula])
                    		hist_fitness.append(self.historico_fitness[it][ID_particula])
            		
            		X = [hist_posicoes[it][p1] for it in xrange(len(hist_posicoes))]
            		Y = [hist_posicoes[it][p2] for it in xrange(len(hist_posicoes))]
            		Z = hist_fitness
                        
            		Z_sort   = sort(Z)
            		Y_sort   = [Y[i] for i in argsort(Z)]
            		X_sort   = [X[i] for i in argsort(Z)]    
                
            		fig = figure()
            		ax = fig.add_subplot(111, projection='3d')
            		ax.scatter(X_sort,Y_sort,Z_sort,c=Z_sort,cmap=cm.coolwarm,zorder=1)
            		arrow = Arrow3D([self.gbest[p1],self.gbest[p1]],[self.gbest[p2],self.gbest[p2]],[max(self.historico_fitness)/4,min(self.historico_fitness)], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
            		ax.add_artist(arrow)
            		ax.yaxis.grid(color='gray', linestyle='dashed')                        
            		ax.xaxis.grid(color='gray', linestyle='dashed')
            		ax.set_zlabel(r"$\quad \Phi $ ",rotation = 180, fontsize = 20)

            		ax.set_zlim([min(self.historico_fitness), max(self.historico_fitness)])
            		ax.set_ylim([self.limite_inferior[p2],self.limite_superior[p2]])
            		ax.set_xlim([self.limite_inferior[p1],self.limite_superior[p1]])

            		if  isinstance(Nome_param, list):
            		    if len(Nome_param) == 0:
            			ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
            			ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)
            		    elif isinstance(Unid_param,list):
            			if len(Unid_param) == 0:
            			    ax.set_ylabel(Nome_param[p2], fontsize = 20)
            			    ax.set_xlabel(Nome_param[p1], fontsize = 20)
            		    elif Unid_param == None:
            			ax.set_ylabel(Nome_param[p2], fontsize = 20)
            			ax.set_xlabel(Nome_param[p1], fontsize = 20)
            		    else:    
            			ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize = 20)
            			ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize = 20)
            		else:
            		    ax.set_ylabel(r"$x_{%0.0f}$ "%(p2+1), fontsize = 20)
            		    ax.set_xlabel(r"$x_{%0.0f}$ "%(p1+1), fontsize = 20)		
            		
            		if len(azim) == 1:
            		    ax.azim = azim[0]
            		else:
            		    ax.azim = azim[pos]    
            		
            		if len(elev) == 1:
            		    ax.elev = elev[0]
            		else:
            		    ax.elev = elev[pos]	    
            		ax.w_zaxis.set_major_formatter(formatter)
            		fig.savefig(dir3+'Funcao_objetivo'+'_x'+str(p1+1)+'_x'+str(p2+1)+'_it'+numeracao+'.png')
            		close()

                    p2+=1

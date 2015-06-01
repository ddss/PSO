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

Reinitialization velocity for HPSO:
- TVVr-linear         - time variating reinitialization velocity (linear)
- Constante
"""

# -------------------------------------------------------------------------------
# IMPORTAÇÃO DE PACOTES (packages import)
# -------------------------------------------------------------------------------
# Sistema
import sys
reload(sys)
sys.setdefaultencoding("utf-8") # Forçar o sistema utilizar o coding utf-8

# Importação de pacotes para cálculos
import os
from time import sleep, ctime, time
import sys
from threading import Thread, Lock, BoundedSemaphore

from numpy import random, min, argmin, copy, matrix, multiply, random, size, min, \
    max, copysign, ones, mean, std, sqrt, cos, pi, sort, argsort, savetxt
from scipy.misc import factorial
from math import isnan
import codecs

# Importação de pacotes para gráficos
from matplotlib import use, cm, ticker

use('Agg')

from matplotlib.pyplot import figure, axes, plot, subplot, xlabel, ylabel, \
    title, legend, savefig, xlim, ylim, close, gca, hist
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import Ellipse, FancyArrowPatch

from warnings import warn

from auxiliares import Validacao_Diretorio, Arrow3D
# -------------------------------------------------------------------------------
# ÍNICIO (beginning)
# -------------------------------------------------------------------------------

class Particula(Thread):
    # Classe que define a partícula, cada partícula é definida como uma Thread
    velocidade = 0
    fitness = 0
    posicao = 0
    Vo = 0
    phi = 0
    qsi = 0
    pbest = 0

    def __init__(self, FO, w, C1, C2, Vmax, Vreinit, num_parametros,  metodo, limite_superior,
                 limite_inferior, ID_particle, num_particulas, args_model=[]):

        global vetor_posicoes, vetor_fitness, vetor_velocidades, vetor_pbest

        # TODO: reestruturar classe partícula: verificar necessidade de atributos self.
        Thread.__init__(self)

        # Atribuição da Função objetivo
        self.FO = FO

        # Atribuição de variáveis self
        self.metodo = metodo
        self.Vmax = Vmax
        self.V_reinit = Vreinit
        self.args_model = args_model

        self.w = w
        self.C1 = C1
        self.C2 = C2

        self.__num_parametros = num_parametros
        self.__num_particulas = num_particulas

        self.ID_particle = ID_particle

        # Limites superiores
        self.__limite_superior = limite_superior
        self.__limite_inferior = limite_inferior

        # Determinação da posição atual da partícula, velocidade inicial (PSO) e pbest 

        self.posicao = copy(vetor_posicoes[self.ID_particle]).tolist() # Posição da partícula
        self.Vo = copy(vetor_velocidades[self.ID_particle]).tolist()   # Velocidade na iteração anterior

        # Busca pela melhor posição desta partícula:
        # pbest é uma lista de listas
        # 0 - posição
        # 1 - fitness em pbest
        self.pbest = vetor_pbest[0][self.ID_particle] # Posição para a qual a partícula obteve o melhor fitness

    def Fitness_Particula(self):
        # Cálculo do Fitness da partícula
        Thfitness = self.FO(self.posicao, self.args_model)
        Thfitness.start()
        Thfitness.join()

        self.fitness = Thfitness.result

    def Execucao_PSO(self):
        # Método para cálculo do particle swarm optimization (PSO) e suas variações
        global gbest

        # Cálculo da velocidade
        R1 = random.uniform(0, 1, size(self.posicao))  # Termo de aceleração aleatório
        R2 = random.uniform(0, 1, size(self.posicao))  # Termo de aceleração aleatório

        vel1 = [Vo * (self.w) for Vo in self.Vo]
        vel2 = [self.C1 * R1[j] * (self.pbest[j] - self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel3 = [self.C2 * R2[j] * (gbest[j] - self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel4 = [vel1[j] + vel2[j] + vel3[j] for j in xrange(self.__num_parametros)]

        if self.metodo.inercia != 'TVIW-Adaptative-VI':
            # Verificação se a velocidade calculada é menor que Vmax, para cada dimensão
            self.velocidade = [0] * self.__num_parametros

            for i in xrange(self.__num_parametros):
                vel5 = min((abs(vel4[i]), self.Vmax[i]))
                aux = copysign(vel5, vel4[i])
                if not isinstance(aux, float):
                    self.velocidade[i] = float(aux)
                else:
                    self.velocidade[i] = aux

        else:
            self.velocidade = vel4

        # Nova posição
        self.posicao = [self.posicao[j] + self.velocidade[j] for j in xrange(self.__num_parametros)]

        # Validação de o valor da posição está contido dentro da região da restrição em caixa (limites originais do problema)
        if self.metodo.restricao == True:
            for i in xrange(self.__num_parametros):
                if self.posicao[i] > self.__limite_superior[i]:
                    self.posicao[i] = self.__limite_superior[i]
                    self.velocidade[i] = -0.10 * self.velocidade[i]

                elif self.posicao[i] < self.__limite_inferior[i]:
                    self.posicao[i] = self.__limite_inferior[i]
                    self.velocidade[i] = -0.10 * self.velocidade[i]

        self.Fitness_Particula()

    def Execucao_HPSO(self):
        global gbest

        # Cálculo da velocidade

        R1 = random.uniform(0, 1, size(self.posicao))  # Termo de aceleração aleatório
        R2 = random.uniform(0, 1, size(self.posicao))  # Termo de aceleração aleatório

        vel1 = [self.C1 * R1[j] * (self.pbest[j] - self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel2 = [self.C2 * R2[j] * (gbest[j] - self.posicao[j]) for j in xrange(self.__num_parametros)]
        vel3 = [vel1[j] + vel2[j] for j in xrange(self.__num_parametros)]

        self.velocidade = [0] * self.__num_parametros
        for i in xrange(self.__num_parametros):

            if vel3[i] == 0:
                if random.uniform(0, 1) < 0.5:
                    vel3[i] = random.uniform(0, 1) * self.V_reinit[i]
                else:
                    vel3[i] = -random.uniform(0, 1) * self.V_reinit[i]

            vel4 = min((abs(vel3[i]), self.Vmax[i]))

            self.velocidade[i] = copysign(vel4, vel3[i])

        # Nova posição
        self.posicao = [self.posicao[j] + self.velocidade[j] for j in xrange(self.__num_parametros)]

        # Validação de o valor da posição está conttido dentro da região da restrição em caixa (limites originais do problema)
        if self.metodo.restricao == True:
            for i in xrange(self.__num_parametros):
                if self.posicao[i] > self.__limite_superior[i]:
                    self.posicao[i] = self.__limite_superior[i]
                elif self.posicao[i] < self.__limite_inferior[i]:
                    self.posicao[i] = self.__limite_inferior[i]

        self.Fitness_Particula()

    def Armazenamento(self):

        global vetor_posicoes, vetor_velocidades, vetor_fitness, vetor_pbest, best_fitness, gbest, Controle_variaveis

        # Método para verificar se haverá mudança em gbest

        # Início da Região Crítica - Compartilhada por todas as threads
        Controle_variaveis.acquire()

        # Atualização dos vetores (históricos / Memória)
        vetor_posicoes[self.ID_particle] = self.posicao
        vetor_velocidades[self.ID_particle] = self.velocidade
        vetor_fitness[self.ID_particle] = self.fitness

        # atualizando pbest
        # pbest é uma lista de listas
        # 0 - posição
        # 1 - fitness em pbest
        if self.fitness < vetor_pbest[1][self.ID_particle]:
            vetor_pbest[0][self.ID_particle] = self.posicao

        # Teste para verificar mudança no mínimo global:
        if self.metodo.gbest == 'Particula':
            if self.fitness < best_fitness:
                best_fitness = self.fitness
                if self.metodo.busca == 'Otimo':
                    gbest = self.posicao  # O valor de gbest é apenas atualizado, caso esteja se buscando o ponto ótimo

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

        total_particulas_atendidas = total_particulas_atendidas + 1

        if (total_particulas_atendidas) == self.__num_particulas:
            Controle_Iteracao.release()

        Controle_Total_Threads.release()

        Controle_Particula.release()


class Metodo:

    def __init__(self, metodo):
        '''
        Classe para transformar o argumento ``metodo`` do PSO, em atributos. Além de definir os valores default dos métodos.
        
        Chaves disponíveis: ``busca`` , ``algoritmo`` , ``inercia`` , ``aceleracao`` , ``Vreinit`` , ``restricao`` , ``gbest``, ``parada``
        
        =================================
        Conteúdos disponíveis por chaves:
        =================================
        
        **Chaves obrigatórias**
        
        * ``busca``: ``Otimo`` ou ``Regiao``
        
        **Chaves não obrigatórias** (com valores default)
        
        * ``algoritmo``: ``PSO`` ou ``HPSO``
            
        * ``inercia``: ``Constante`` ou ``TVIW-linear`` ou ``TVIW-Adaptative-VI`` ou ``TVIW-random``
        
        * ``aceleracao``: ``Constante`` ou ``TVAC``

        * ``Vreinit``: ``Constante`` ou ``TVVr-linear``
        
        * ``restricao``: ``True`` ou ``False``

        * ``gbest``: ``Particula`` ou ``Enxame``

        * ``parada``: ``itmax`` ou ``desviorelativo`` ou ``evolucaogbest``
                
        ================
        Valores default:
        ================
        
        **Chaves obrigatórias**

        * ``busca``: define o método de busca. Conteúdos disponíveis são: ``Otimo`` ou ``Regiao``

        **Chaves não obrigatórias** 
        
        * ``algoritmo``: define o algoritmo a ser utilizado. Se ``busca`` for ``Regiao``, então é definido ``HPSO``. Caso contrário ``PSO``
            
        * ``inercia``: define o método do peso de inércia. Se ``algoritmo`` é ``PSO`` e ``busca`` è ``Regiao``, então, o método de inérci é ``TVIW-linear``. Caso contrário, ``Constante``
        
        * ``aceleracao``: define o método para fatores de aceleração. Se ``algoritmo`` é ``PSO``, ``aceleração`` é ``Constante``. Caso ``algoritmo`` seja ``HPSO`` e ``busca`` seja ``Otimo``, então ``TVAC``. Se ``busca`` for ``Regiao``, então ``Constante``.

        * ``Vreinit``: método para definir o comportamento da reinicialização do HPSO. Caso não definido é utilizado o ``TVVr-linear``
        
        * ``restricao``: define se os limites serão utilizados como restrições. Caso não definido é assumido o valor lógico ``True``
        
        * ``gbest``: define como a atualização de ``gbest`` (ponto ótimo) é realizada. Se o método de busca é ``Otimo``, então ``Particula`` . Caso o método de busca seja ``Regiao``, então ``Enxame``.

        * ``parada``: define como o algoritmo de PSO deve finalizar sua execução
        '''
        # -------------------------------------------------------------------------------
        # VALIDAÇÃO
        # -------------------------------------------------------------------------------
        # Métodos disponíveis:
        # TODO: modificar nome do método 'TVIW-Adaptative-VI' para TVIW-Adaptive-vel
        self._metodosdisponiveis = {'busca':       ['Otimo', 'Regiao'],
                                     'algoritmo':  ['PSO', 'HPSO'],
                                     'inercia':    ['TVIW-linear', 'TVIW-random', 'Constante', 'TVIW-Adaptative-VI'],
                                     'aceleracao': ['TVAC', 'Constante'],
                                     'Vreinit':    ['TVVr-linear', 'Constante'],
                                     'restricao':  [True,False],
                                     'gbest':      ['Particula','Enxame'],
                                     'parada':     ['itmax','desviorelativo','evolucaogbest']}

        # Validações:
        # O método de busca é obrigatório ser completamente definido
        if 'busca' not in metodo.keys():
            raise NameError('A definição do método de busca é obrigatória')

        if metodo['busca'] not in self._metodosdisponiveis['busca']:
            raise ValueError('É necessário definir um método de busca. Valores disponíveis: '
                             +', '.join(self._metodosdisponiveis['busca'])+'.')

        # validação dos outros métodos
        for key in metodo.keys():
            # Teste para validar se as chaves definidas existem
            if key not in self._metodosdisponiveis.keys():
                raise NameError(u'Keyword {}'.format(key)+u' is not available! Available keywords: '+
                                ', '.join(self._metodosdisponiveis.keys())+u'.')

            if metodo[key] is not None:
                # Teste para validar se o conteúdo são strings (exceto para a chave restrição)
                if not isinstance(metodo[key],str) and key != 'restricao':
                    raise TypeError('Para a chave {} o valor {} deve ser um string'.format(key,metodo[key]))

                # Teste para validar se o conteúdo da chave restrição é bool
                if not isinstance(metodo[key],bool) and key == 'restricao':
                    raise TypeError('Para a chave {} o valor {} deve ser um bool'.format(key,metodo[key]))

                # Teste para validar se o conteúdo definido está disponível
                if metodo[key] not in self._metodosdisponiveis[key]:
                    raise NameError(u'The value {} is not available for key {}!'.format(metodo[key], key)+u' Available values: '+
                                ', '.join(self._metodosdisponiveis[key])+u'.')
        # -------------------------------------------------------------------------------
        # ATRIBUTOS
        # -------------------------------------------------------------------------------
        # Atributos cujos valores default que independem de outros:
        # retrições são por default, ativas
        self.restricao = metodo.get('restricao') if metodo.get('restricao') is not None else True
        # critério de parada é, por default, itmax
        self.parada = metodo.get('parada') if metodo.get('parada') is not None else self._metodosdisponiveis['parada'][0]

        # Atributos cujos valores default que dependem de outros, serão inicializados pelo método InicializacaoDefault:
        self.busca = metodo.get('busca')
        self.algoritmo = metodo.get('algoritmo')
        self.inercia = metodo.get('inercia')
        self.aceleracao = metodo.get('aceleracao')
        self.Vreinit = metodo.get('Vreinit')
        self.gbest = metodo.get('gbest')

        # -------------------------------------------------------------------------------
        # INICIALIZAÇÃO
        # -------------------------------------------------------------------------------
        self.InicializacaoDefault()

    def InicializacaoDefault(self):
        """
        Inicialização dos valores default e validações adicionais.
        """
        # ALGORITMO
        if self.algoritmo is None:
            # se o método de busca for região, então é usado o HPSO, senão PSO
            if self.busca == self._metodosdisponiveis['busca'][1]: # regiao
                self.algoritmo = self._metodosdisponiveis['algoritmo'][1] # HPSO
            else:
                self.algoritmo = self._metodosdisponiveis['algoritmo'][0] # PSO

        # INÉRCIA
        if self.inercia is None:
            # se está sendo utilizado o PSO e a busca é para um ponto ótimo, será utilizado o peso de inércia TVIW-linear, caso contrário Constante
            if (self.algoritmo == self._metodosdisponiveis['algoritmo'][0]) and (self.busca == self._metodosdisponiveis['busca'][0]):
                    self.inercia = self._metodosdisponiveis['inercia'][0] # TVIW-linear, conforme apresentado por [1] e [4] -> standard particle swarm optimization
            # se a busca é para uma região e não se está usando HPSO, o peso de inercia será constante
            elif (self._metodosdisponiveis['busca'][1]) and (self.algoritmo != self._metodosdisponiveis['algoritmo'][1]):
                self.inercia = self._metodosdisponiveis['inercia'][2]

        # se está se usando HPSO, o peso de inércia não deve ser definido
        elif (self.inercia in self._metodosdisponiveis['inercia']) and (self.algoritmo == self._metodosdisponiveis['algoritmo'][1]):
            raise ValueError('O peso de inercia é apenas suportado no {}. Está sendo utilizado {}, utilize None'.format(self._metodosdisponiveis['algoritmo'][0],self._metodosdisponiveis['algoritmo'][1]))

        # ACELERAÇÃO
        if self.aceleracao is None:  # Método para cálculo de C1 e C2
            # se o algoritmo for o PSO, a aceleração será constante
            if self.algoritmo == self._metodosdisponiveis['algoritmo'][0]:
                self.aceleracao = self._metodosdisponiveis['aceleracao'][1] # Constante
            # se o algoritmo for o HPSO, a aceleração será TVAC (Busca ótimo) ou Constante (Busca Região)
            elif self.algoritmo == self._metodosdisponiveis['algoritmo'][1]: # HPSO
                if self.busca == self._metodosdisponiveis['busca'][0]: # Busca: Ótimo
                    self.aceleracao = self._metodosdisponiveis['aceleracao'][0] # C: TVAC
                elif self.busca == self._metodosdisponiveis['busca'][1]: # Busca: Regiao
                    self.aceleracao = self._metodosdisponiveis['aceleracao'][1] # C: Constante

        # VELOCIDADE DE REINICIALIZAÇÃO
        # se o algoritmo for HPSO, a velocidade de reinicialiação é TVVr-linear
        if self.algoritmo == self._metodosdisponiveis['algoritmo'][1]: # HPSO
            if self.Vreinit is None:
                self.Vreinit = self._metodosdisponiveis['Vreinit'][0] # TVVr-linear

        # O método Vreinit só é aplicável para o HPSO
        if (self.Vreinit is not None) and (self.algoritmo != self._metodosdisponiveis['algoritmo'][1]):
            raise ValueError('Vreinit method is only applied with {} algorithm. Do not use it'.format(self._metodosdisponiveis['algoritmo'][1]))

        # ATUALIZAÇÃO DE GBEST
        if self.gbest is None:
            # se está se buscando um ponto ótimo, gbest é atualizado pelas partículas
            if self.busca == self._metodosdisponiveis['busca'][0]: # Otimo
                self.gbest = self._metodosdisponiveis['gbest'][0] # Particula
            # se está se buscando uma região, gbest é atualizado pelo enxame
            elif self.busca == self._metodosdisponiveis['busca'][1]: # Regiao
                self.gbest = self._metodosdisponiveis['gbest'][1] # Enxame

        # PARADA
        if self.parada == self._metodosdisponiveis['parada'][1] and self.algoritmo == self._metodosdisponiveis['algoritmo'][1]:
            raise ValueError('O critério de parada {} não pode ser usado com o algoritmo {}'.format(self.parada,self.algoritmo))

class PSO:
    def __init__(self, limite_superior, limite_inferior, metodo={'busca': 'Otimo'}, Num_particulas=30, itmax=2000,
                 **kwargs):
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
            
             >>> from threading import Thread
             >>>
             >>> class NomeFuncaoObjetivo(Thread):
             >>>    result = 0
             >>>    def __init__(self,parametros,argumentos_extras):
             >>>        Thread.__init__(self)
             >>>        self.param = parametros
             >>>        self.args  = argumentos_extras
             >>>
             >>>    def run(self):
             >>>        #Calculos utilizado self.param (e self.args, opcionalmente)
             >>>
             >>>        self.result = float() # Resultado da função objetivo (deve ser um float)
        
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
        
            >>> from threading import Thread
            >>>
            >>> class FO(Thread):
            >>>    result = 0
            >>>    def __init__(self,param,args):
            >>>        Thread.__init__(self)
            >>>        self.x = param
            >>>
            >>>    def run(self):
            >>>
            >>>        self.result =  self.x**2

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
                
                * ``PSO`` (Particle Swarm Optimization) (conteúdo, string): algoritmo de PSO com peso de inércia, fatores de aceleração
                * ``HPSO`` (Self-Organizing Hierarchical Particle Swarm Optimizer) (conteúdo, string): executa um método de PSO assumindo peso de inércia zero e reiniciando as partículas quando suas velocidades são zero.
            
            * ``inercia``: define como o algoritmo calcula o peso de inércia
            
                * ``Constante`` (conteúdo, string): o peso de inércia é assumido constante e não varia ao longo das iterações (Vide [1] e [4])
                * ``TVIW-linear`` (*linear time varying inertia weight*) (conteúdo, string): o peso de inércia varia linearmente ao longo das iterações (VIde [4])
                * ``TVIW-Adaptative-VI`` (*adaptive time varying inertia weight based on velocity*) (conteúdo, string): o peso de inércia e ajustado no algoritmo para forçar o perfil de velocidade a um comportamento ideal (Vide [8])
                * ``TVIW-random`` (*random time varying inertia weight*) (conteúdo, string): o peso de inércia é variado aleatoriamente entre 0 e 1 (Vide [4])
            
            * ``aceleracao`` (chave):  define como o algoritmo calcula as constantes de aceleração
            
                * ``Constante`` (conteúdo, string):  os fatores de aceleração são assumidos constantes e não variam ao longo das iterações (Vide [4])
                * ``TVAC`` (*time varying acceleration constant*) (conteúdo, string):  os constantes de aceleraçao variam linearmente ao longo das iterações (Vide [10])

            * ``restriçao`` (chave): define se os limites de busca serão utilizados como restrições do problema de otimização
            
                * ``True`` (conteúdo, bool): os limites são utilizados como restrições
                * ``False`` (conteúdo, bool): os limites não são utilizados como restrições 
                
            * ``gbest`` (chave): define a forma como o ponto ótimo (gbest) é atualizado ao longo das iterações
            
                * ``Particula`` (conteúdo, string): gbest será atualizado ao longo da iteração pelas partículas
                * ``Enxame`` (conteúdo, string): gbest será atualizado ao final de cada iteração
            
            * ``Vreinit`` (chave):  define a forma de tratar a velocidade de reinicilização no algoritmo de HPSO (Vide [10])
            
                * ``TVVr-linear`` (time varying reinitialization velocity) : a velocidade de reinicialização decresce linearmente ao longo das iterações
                * ``Constante`` a velocidade é mantida constante

            * ``parada`` (chave): define o critério de parada do algoritmo

                * ``itmax``: o algoritmo para se execução quando atinge o número máximo de iterações
                * ``desviorelativo``: o algoritmo para sua execução quando o desvio relativo dos valores de função objetivo obtidos pelas partículas e o valor de gbest (ponto ótimo) é alcançado. Útil para evitar excessivo número de iterações. Caso o algorimo seja o HPSO, este critério não deve ser utilizado.
                * ``evolucaogbest``: o algoritmo para sua execução quando o gbest não está sendo melhorado. Útil para evitar excessivo número de iterações.
                
        Na definição do método, apenas a chave ``busca`` é obrigatória. Se algumas das restantes não for definida, será assumido valores default. (vide a documentação da classe ``Metodo``)
        
        Caso o método seja omitido, na chamada da função PSO, conforme Exemplo 1, será utilizado o método: ::
        
        {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'Constante','restricao':True,'gbest':'Particula','parada':'itmax'}
        
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
                >>> metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','restricao':False,'gbest':'Particula'}
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
        * ``args_model``: lista que possui argumentos extras a serem passados para a função objetivo. Seu valor default é uma lista vazia.
        * ``n_historico`` (float): tamanho máximo do histório das iterações a ser salvo pelo algoritmo (limita a quantidade de informações que são salvas). Default: min(itmax,500)
        * ``itmin`` (float): número mínimo de iterações a ser realizado pelo algoritmo (útil para evitar convergência prematura, caso o critério de convergência esteja ativado). Default: 1
        * ``n_desempenho`` (float): número de amostragens a serem realizadas, para indicação do desempenho ao algoritmo. Default: 20

        ==========
        Exemplo 3
        ==========
        
        O **exemplo 2** será resolvido novamente, definindo os valores do peso de inércia (``w``) entre 0.9 e 0.3, fator de aceleração individual (``C1``) entre 2.5 e 0.5 e fator de aceleração social (``C2``) entre 0.5 e 2.5. Observe que ``w`` e ``C1`` decrescem, enquanto ``C22`` aumenta. ::
        
                >>> from funcaoobjetivo import FO
                >>> sup  = [10.]  # limite superior de busca para x
                >>> inf  = [-10.] # limite inferior de busca para x
                >>> metodo = {'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear','aceleracao':'TVAC','restricao':False,'gbest':'Particula'}
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
        # TODO: Implementar critérios de parada

        # -------------------------------------------------------------------------------
        # VALIDAÇÃO
        # -------------------------------------------------------------------------------
        # VALIDAÇÕES DE ENTRADAS
        # Os limites inferior e superior devem ser listas
        if not isinstance(limite_inferior, list) or not isinstance(limite_superior, list):
            raise TypeError('Os limites inferiores e superiores devem ser listas.')

        # Os limites inferior e superior devem ter a mesma dimensão
        if len(limite_inferior) != len(limite_superior):
            raise NameError('Os limites_superior e limite_inferior devem ter a mesma dimensão')

        # VALIDAÇÕES DE KEYWORDS:
        # keywords disponíveis com seus respectivos tipos
        # args_model não é validado, pois depende da função sendo minimizada.
        keydisponiveis = {'posinit_sup':list, 'posinit_inf':list,
                          'w':list, 'C1':list, 'C2':list,
                          'Vmax':list, 'Vreinit':list,
                          'otimo':list, 'deltaw':float,'args_model':None,
                          'itmin':int, 'n_historico':int}

        for key in kwargs.keys():
            # validação se a keyword existe
            if not key in keydisponiveis.keys():
                raise NameError('Keyword: {} not available!'.format(key)+ ' Available keywords: '+','.join(keydisponiveis.keys())+'.')
            # validação de o tipo está correto
            if kwargs.get(key) is not None and keydisponiveis[key] is not None:
                if not isinstance(kwargs.get(key),keydisponiveis[key]):
                    raise TypeError('A chave {} deve ter como conteúdo {}'.format(key,keydisponiveis[key]))

        # O limite de inicialização, caso definido, deve ser uma lista e ter meesma dimensão
        # do limite_inferior e limite_superior
        if kwargs.get('posinit_sup') is not None:
            # teste de dimensão
            if len(kwargs.get('posinit_sup')) != len(limite_superior):
                raise ValueError('O limite de inicialização posinit_sup deve ter a mesma dimensão de limite_superior e limite_inferior')

        # O limite de inicialização, caso definido, deve ser uma lista e ter meesma dimensão
        # do limite_inferior e limite_superior
        if kwargs.get('posinit_inf') is not None:
            # teste de dimensão
            if len(kwargs.get('posinit_inf')) != len(limite_superior):
                raise ValueError('O limite de inicialização posinit_inf deve ter a mesma dimensão de limite_superior e limite_inferior')

        # teste para avaliar se o limite superior é menor do que  o inferior.
        for i in xrange(len(limite_superior)):
            if limite_inferior[i] >= limite_superior[i]:
                raise ValueError('O limite inferior deve ser menor do que o limite superior para todas as dimensões')

        # teste para avaliar se o ponto focal tem mesmo tamanho do número de parâmetros
        if kwargs.get('otimo') is not None:
            if len(kwargs.get('otimo')) != len(limite_superior):
                raise ValueError('O ponto ótimo deve ter a mesma dimensão do limite_superior e limite_inferior')

        # -------------------------------------------------------------------------------
        # INICIALIZAÇÃO DO MÉTODO
        # -------------------------------------------------------------------------------
        self.metodo = Metodo(metodo)

        # -------------------------------------------------------------------------------
        # INICIALIZAÇÃO DE ATRIBUTOS CUJOS VALORES DEFAULT SÃO PRÉ-DEFINIDOS
        # -------------------------------------------------------------------------------
        # número máximo de iterações
        self.itmax = int(itmax + 1)  # A iteração 0 é a inicialização
        # número de partículas
        self.Num_particulas = int(Num_particulas)
        # número de parâmetros:  determinado pela dimensão do limite superior
        self.Num_parametros = len(limite_superior)

        # limite inferior de busca
        self.limite_inferior = [float(lim) for lim in limite_inferior]  # Lista
        # limite superior de busca
        self.limite_superior = [float(lim) for lim in limite_superior]  # Lista
        # ponto focal do algoritmo - quando o método de busca é Regiao ele subsitui gbest
        self.foco = kwargs.get('otimo')

        # TRATAMENTO DAS KWARGS:
        # Valores default das kwargs (Aquelas que não dependem do método escolhido):
        # posições de inicialização do algoritmo - limite superior:
        self.posinit_sup = kwargs.get('posinit_sup') if kwargs.get('posinit_sup') is not None else limite_superior
        # posições de inicialização do algoritmo - limite inferior:
        self.posinit_inf = kwargs.get('posinit_inf') if kwargs.get('posinit_inf') is not None else limite_inferior
         # Parâmetro de seleção para o algoritmo PSO-TVIW-Adaptative-VI | Default 0.1, conforme recomendação de [8]:
        self.deltaw = kwargs.get('deltaw') if kwargs.get('deltaw') is not None else 0.1
        # Argumentos extras a serem enviados para o modelo:
        self.args_model = kwargs.get('args_model') if kwargs.get('args_model') is not None else []
        # TODO: Implementar itmin
        # Número mínimo de iterações, caso não definido será 1
        itmin = kwargs.get('itmin') if kwargs.get('itmin') is not None else 1
        # tamanho do histórico
        self.n_historico = kwargs.get('n_historico') if kwargs.get('n_historico') is not None else min([500, itmax])
        # identificação do desempenho
        self.n_desempenho = kwargs.get('n_desempenho') if kwargs.get('n_desempenho') is not None else min([20,itmax])
        # -------------------------------------------------------------------------------
        # VALIDAÇÕES ADICIONAIS
        # -------------------------------------------------------------------------------
        if (self.itmax - 1) == 1 and \
            (self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][0] or
            self.metodo.aceleracao == self.metodo._metodosdisponiveis['aceleracao'][0]):
            raise ValueError('Para o método de inércia {}'.format(self.metodo._metodosdisponiveis['inercia'][0])+
                             ' e método de aceleraçãode {}'.format(self.metodo._metodosdisponiveis['aceleracao'][0])+
                             ', o número de iterações deve ser, no mínimo, 2')

        if self.metodo.busca == self.metodo._metodosdisponiveis['busca'][1] and self.foco is None:
                raise ValueError('Dado que o método de busca é {}'.format(self.metodo._metodosdisponiveis['busca'][1])+
                                 ' necessário informar o valor de um ponto ótimo da função. keyword: otimo')

        if self.n_historico >= 2000:
            warn('Grandes históricos (n_historicos) podem levar a um alto consumo de memória e reduzir o desempenho')

        if self.n_historico > itmax:
            raise ValueError('O histórico não pode ser maior do que o número máximo de iterações')

        if self.n_desempenho > itmax:
            raise ValueError('O n_desempenho deve ser, no máximo, o número de iterações'.format(itmax))

        if self.n_desempenho > 100:
            warn('Muitas amostragens (n_desempenho) pode levar a um alto consumo de memória e reduzir o desempenho')

        # -------------------------------------------------------------------------------
        # INICIALIZAÇÃO DE ATRIBUTOS CUJOS VALORES DEFAULT DEPENDEM DO MÉTODO
        # -------------------------------------------------------------------------------
        # default: None
        w = kwargs.get('w')    # Peso de inércia (definido None, pois seu valor default depende do método) (usado no PSO)
        C1 = kwargs.get('C1')  # Fator de aceleração do coeficiente individual (definido None, pois seu valor default depende do método)
        C2 = kwargs.get('C2')  # Fator de aceleração do coeficiente social (definido None, pois seu valor default depende do método)
        Vmax = kwargs.get('Vmax')  # Velocidade máxima do algoritmo
        Vreinit = kwargs.get('Vreinit')  # Velocidade de reiniciação (usada do HPSO): Vreinit must be a list. Structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]

        # inicialização dos valores default de w, C1 e C2 e criação dos atributos
        self.__defaultCoeficientes(w, C1, C2)
        self.__defaultVelocidades(Vmax, Vreinit)

        # -------------------------------------------------------------------------------
        # INICIALIZAÇÃO DAS LISTAS DE HISTÓRICO
        # -------------------------------------------------------------------------------
        # lista com os índices das iterações nas quais o desempenho será avaliado
        self.index_desempenho = [(int((self.itmax-1)/self.n_desempenho))*k for k in xrange(0,self.n_desempenho+1)]
        # vetores baseados nas amostragens do desempenho
        self.media_fitness = [0] * (self.n_desempenho+1)
        self.media_velocidade = [0] * (self.n_desempenho+1)
        self.velocidade_ideal = [0] * (self.n_desempenho+1)
        self.desvio_fitness = [0] * (self.n_desempenho+1)
        self.historico_w = [0] * (self.n_desempenho+1)
        self.historico_best_fitness = [0] * (self.n_desempenho+1)

        # vetores baseados nos n_historico - salvam os n_historico últimos valores
        self.historico_fitness = [0]* self.n_historico
        self.historico_posicoes = [0]* self.n_historico

    def __defaultCoeficientes(self, w, C1, C2):
        '''
        Subrotina para definição dos valores default para o PESO DE INÉRCIA E FATOR DE ACELERAÇÃO
        
        ===============        
        Peso de inércia
        ===============
        
        * Se o método para cálculo do peso de inércia é ``Constante``, w = [0.9].
        * Se o método para cálculo do peso de inércia é ``TVIW-linear``, ``w = [0.9,0.4]``
        * Se o método para cálculo do peso de inércia é  ``TVIW-random``, ``w = None``
        * Se o método para cálculo do peso de inércia é  ``TVIW-Adaptative-VI``, ``w = [0.9,0.3]``
        
        ================================
        Fator de aceleração individual
        ================================
        
        * Se o método para cálculo dos fatores de aceleração é ``Constante``, ``C1 = [2.0]``.
        * Se o método para cálculo dos fatores de aceleração é ``TVAC``, ``C1 = [2.5,0.5]`` . 
        * Se o método para cálculo do peso de inércia é ``TVIW-Adaptive-VI``, ``C1 = [1.49]`` . 

        ================================
        Fator de aceleração social
        ================================
        
        * Se o método para cálculo dos fatores de aceleração é ``Constante`` ``C1 = [2.0]``.
        * Se o método para cálculo dos fatores de aceleração é ``TVAC``, ``C1 = [0.5,2.5]`` . 
        * Se o método para cálculo do peso de inércia é ``TVIW-Adaptive-VI``, ``C1 = [1.49]`` . 
        
        **Observe que NEM todas as combinações de métodos possuem valores *default* de parâmetros. Use com cautela**
        '''
        # -------------------------------------------------------------------------------
        # PESO DE INÉRCIA
        # -------------------------------------------------------------------------------
        if w is None:
            if self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][0]: # TVIW-linear
                self.w = [0.9,0.4]  # Conforme valor recomendado por [4]
            elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][1]: # TVIW-random
                self.w = [None]*2  # Valor não será necessário. W é aleatório
            elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][2]: # Constante
                self.w = [0.9]*2 # Conforme valor recomendado por [4]
            elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][3]: #TVIW-Adaptive-vel
                self.w = [0.9,0.3]  # Conforme valor recomendado por [8]
            elif self.metodo.inercia is None:
                self.w = [None]*2
        else:
            if self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][0] or self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][3]:
                if len(w) == 2:
                    self.w = w
                else:
                    raise ValueError('w deve ser uma lista de 2 valores, visto que o método para w é {}'.format(self.metodo.inercia))

            elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][1]: # TVIW-random
                if w is None:
                    self.w = [None]*2  # Valor não será necessário. W é aleatório
                else:
                    raise ValueError('w deve ser None, pois varia aleatoriamente, visto que o método para w é {}'.format(self.metodo.inercia))

            elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][2]: # Constante
                if len(w) == 1:
                    self.w = w*2
                else:
                    raise ValueError('w deve ser uma lista de 1 valor, visto que o método para w é {}'.format(self.metodo.inercia))
            elif self.metodo.inercia is None:
                self.w = [None]*2

        # -------------------------------------------------------------------------------
        # COEFICIENTE DE ACELERAÇÃO DO TERMO COGNITIVO
        # -------------------------------------------------------------------------------
        if C1 is None:
            if (self.metodo.algoritmo == self.metodo._metodosdisponiveis['algoritmo'][0]) or\
                    (self.metodo.algoritmo == self.metodo._metodosdisponiveis['algoritmo'][1]):
                if self.metodo.aceleracao == self.metodo._metodosdisponiveis['aceleracao'][0]: # TVAC
                    self.C1 = [2.5,0.5]  # Default 2.5, conforme recomendado por [10]
                elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][3]: # inercia: TVIW-Adaptive-vel
                    self.C1 = [1.49]*2
                else:
                    self.C1 = [2.0]*2  # Default 2, conforme recomendado por [4]
        else:
            if self.metodo.aceleracao == self.metodo._metodosdisponiveis['aceleracao'][0]: # TVAC
                if len(C2) == 2:
                    self.C1 = C1
                else:
                    raise ValueError('C1 deve ser uma lista de 2 valores para usar o método {}'.format(self.metodo.aceleracao))
            else:
                if len(C1) == 1:
                    self.C1 = C1*2
                else:
                    raise ValueError('C1 deve ser uma lista de 1 valor para usar o método {}'.format(self.metodo.aceleracao))

        # ------------------------------------------------------------------------------
        # COEFICIENTE DE ACELERAÇÃO DO TERMO SOCIAL
        # ------------------------------------------------------------------------------
        if C2 is None:
            if (self.metodo.algoritmo == self.metodo._metodosdisponiveis['algoritmo'][0]) or\
                    (self.metodo.algoritmo == self.metodo._metodosdisponiveis['algoritmo'][1]):
                if self.metodo.aceleracao == self.metodo._metodosdisponiveis['aceleracao'][0]: # TVAC
                    self.C2 = [0.5,2.5]  # Default 0.5, conforme recomendado por [10]
                elif self.metodo.inercia == self.metodo._metodosdisponiveis['inercia'][3]: # inercia: TVIW-Adaptive-vel
                    self.C2 = [1.49]*2
                else:
                    self.C2 = [2.0]*2  # Default 2, conforme recomendado por [4]
        else:
            if self.metodo.aceleracao == self.metodo._metodosdisponiveis['aceleracao'][0]: # TVAC
                if len(C2) == 2:
                    self.C2 = C2
                else:
                    raise ValueError('C2 deve ser uma lista de 2 valores para usar o método {}'.format(self.metodo.aceleracao))
            else:
                if len(C2) == 1:
                    self.C2 = C2*2
                else:
                    raise ValueError('C2 deve ser uma lista de 1 valor para usar o método {}'.format(self.metodo.aceleracao))

    def __defaultVelocidades(self, Vmax, Vreinit):
        '''
        Subrotina para inicialização das VELOCIDADES MÁXIMAS e de REINICIALIZAÇÃO, incluindo seus valores *default*. Validação dos limites de busca.

        ===================
        Velocidades máximas
        ===================

        * A velocidade máxima é mantida no range do problema

        =============================
        Velocidade de reinicialização
        =============================

        *  Se o método para ``Vreinit`` for ``Constante``, a velocidade de reinicialização é inicializada em Vmax, com ``final_percentage`` igual a 1
        *  Se o método para ``Vreinit`` for ``TVVr-linear``, a velocidade de reinicialização é inicializada em Vmax, com ``final_percentage`` igual a 0.1
        '''
        # ------------------------------------------------------------------------------
        # VELOCIDADE MÁXIMA
        # ------------------------------------------------------------------------------
        if Vmax is None:
            # Conforme [4], Vmax é mantido no limite dinâmico do problema
            self.Vmax = [max([abs(self.limite_inferior[i]), abs(self.limite_superior[i])]) for i in xrange(self.Num_parametros)]
        else:
            if len(Vmax) == self.Num_parametros:
                self.Vmax = Vmax
            else:
                raise ValueError('Vmax deve ter a mesma dimensão do limite_inferior e limite_superior')

        # ------------------------------------------------------------------------------
        # VELOCIDADE DE REINICIALIZAÇÃO - HPSO
        # ------------------------------------------------------------------------------
        if self.metodo.algoritmo == self.metodo._metodosdisponiveis['algoritmo'][1]: # HPSO

            if Vreinit is not None:

                if len(Vreinit) != 2:
                    raise ValueError('Vreinit must be a list with 2 elements. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]')

                if not isinstance(Vreinit[0], list):
                    raise ValueError('The first value of Vreinit list must be another list with the same size of parameters number. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]')

                if len(Vreinit[0]) != self.Num_parametros:
                    raise ValueError('The first value of Vreinit list must be another list with the same size of parameters number. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]')

                if not isinstance(Vreinit[1], float):
                    raise ValueError('The second value of Vreinit list must be a float between 0 and 1. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]')

                if not 0 < Vreinit[1] <= 1:
                    raise ValueError('The second value of Vreinit list must be a float between 0 and 1. Vreinit structure: [[Vreinit_1st_param,Vreinit_2nd_param,...],final_percentage]')

            if self.metodo.Vreinit == self.metodo._metodosdisponiveis['Vreinit'][1]: # Constante
                if Vreinit is None:
                    self.Vreinit = [self.Vmax, 1.0]  # Conforme [10], a velocidade de reinicialização é mantida em Vmax
                else:
                    if Vreinit[1] == 1.0:
                        self.Vreinit = Vreinit
                    else:
                        raise ValueError('The porcentage must be 1 for Vreinit, because the method chosen is {}'.format(self.metodo.Vreinit))

            elif self.metodo.Vreinit == self.metodo._metodosdisponiveis['Vreinit'][0]:
                if Vreinit is None:
                    self.Vreinit = [self.Vmax, 0.1]  # Conforme [10], a velocidade de reinicialização inicial é Vmax e decresce linearmente até 10% deste valor
                else:
                     self.Vreinit = Vreinit

        else:
            self.Vreinit = None  # Não é utilizada pelos demais métodos

        # ------------------------------------------------------------------------------
        # VELOCIDADE DE INICIALIZAÇÃO - TVIW-Adaptivel
        # ------------------------------------------------------------------------------
        self.Vstart = abs((max(self.limite_superior) - min(self.limite_inferior)) / 2.0)
        self.Tend = 0.95 * self.itmax

    def Busca(self, FO, printit=False):
        """
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
        """
        # ------------------------------------------------------------------------------
        # VARIÁVEIS GLOBAIS - Compartilhada com Partícula
        # ------------------------------------------------------------------------------
        #  Controle de threads
        global Controle_FO, Controle_Particula, Controle_Iteracao, Controle_variaveis, total_particulas_atendidas, Controle_Total_Threads
        # variáveis compartilhadas
        global vetor_posicoes, vetor_fitness, vetor_velocidades, vetor_pbest, best_fitness, gbest

        # ------------------------------------------------------------------------------
        # CONTROLE DE THREADS - PARTÍCULAS
        # ------------------------------------------------------------------------------
        # Controles de Threads | Semáforo: número máximo de threads que são executadas ao mesmo tempo
        # limita o número de partículas sendo executadas ao mesmo tempo
        max_Threads_permitido = 50

        # num_Threads é mantido no número de partículas, caso não ultrapasse max_Threads_permitido
        if self.Num_particulas <= max_Threads_permitido:
            num_Threads = self.Num_particulas

        else:
            num_Threads = max_Threads_permitido

        Controle_Particula = BoundedSemaphore(value=num_Threads)  # Semáforo -> Controlar o número máximo de Threads a serem utilizadas pelo Programa (evitar número excessivo) ao mesmo tempo para avaliação das partículas no def Busca
        Controle_FO = BoundedSemaphore(value=num_Threads)  # Semáforo -> Controlar o número máximo de Threads a serem utilizadas pelo Programa (evitar número excessivo) ao mesmo tempo para avaliação da função objetivo no def __init__

        # Definição de Locks (para controle das interações e regiões críticas)
        Controle_Iteracao = Lock()       # Aplicável para evitar que a iteração acabe prematuramente, devido ao não término da execução de Threads
        Controle_variaveis = Lock()      # Aplicável para controlar a região crítica envolvendo as variáveis característica das partículas (posicão, velocidade, fitness e gbest)
        Controle_Total_Threads = Lock()  # Aplicável para controlar a região crítica envolvendo a variável total_particulas_atendidas

        # ------------------------------------------------------------------------------
        # INICIALIAÇÃO DE VARIÁVEIS PARA SALVAR AS INFORMAÇÕES DAS PARTÍCULAS
        # ------------------------------------------------------------------------------
        # Inicializações de listas: posiçoes, velocidades e fitness
        # Estes vetores irão salvar o estado atual das partículas e são globais
        vetor_posicoes = [0]*self.Num_particulas # vetor contendo o histórico das posicões de cada partícula em iterações
        vetor_velocidades = [0]*self.Num_particulas # vetor contendo o histórico das velocidades de cada partícula em iterações
        vetor_fitness = [0]*self.Num_particulas # vetor contendo o histórico do fitness de cada partícula em iterações

        # ----------------------------------------------------------------------------------------
        # INICIALIAÇÃO DO ALGORITMO: vetor_posicoes, vetor_velocidades, vetor_fitness, vetor_pbest
        # -----------------------------------------------------------------------------------------
        threadarray = [] # lista que conterá as threads (fitness)
        ID_FO = 0 # índice para vetores de informações (posicao, fitness) no loop da para armazenamento do valor da
                  #  função objetivo - tem o mesmo significado de ID_particle
        ID_particle = 0 # índice para vetores de informações (posicao, fitness, velocidade) no loop para iniciar a
                        # função objetivo

        while (ID_particle < self.Num_particulas) or (ID_FO < self.Num_particulas):

            # Loop: armazenando o valor da função objetivo calculado pela partícula
            # no vetor_fitness.
            if ID_particle > 0: # apenas quando houverem "partículas" inicializadas. Neste ponto, as partículas são as
                                # threads da função objetivo
                if not (threadarray[ID_FO].isAlive()): # teste para validar de a thread está ativa
                    if size(threadarray[ID_FO].result) > 1: # validação da função objetivo
                        raise ValueError('A função objetivo possui mais de uma dimensão. Verificá-la')
                    else: # se passa no teste, é salva no vetor_fitness
                        vetor_fitness[ID_FO] = float(copy(threadarray[ID_FO].result))
                        ID_FO += 1 # incremento
                        Controle_FO.release() # liberando a posição no semáforo

            # Loop: inicialização da função objetivo
            if ID_particle < self.Num_particulas:

                # Inicialização da posição e velocidade
                pos = []
                vel = []
                for p in xrange(self.Num_parametros):
                    random.seed(ID_particle + p * 2 + 245) # controle de sementes
                    # as posições e velocidades são amostradas de uma uniforme
                    pos.append(random.uniform(self.posinit_inf[p], self.posinit_sup[p]))
                    vel.append(random.uniform(self.posinit_inf[p], self.posinit_sup[p]))
                # Armazenamento dos valores iniciados de posicoes e velocidades
                vetor_posicoes[ID_particle] = copy(pos).tolist()
                vetor_velocidades[ID_particle] = copy(vel).tolist()

                # Avaliação da função objetivo para para posição - fitness da partícula
                Controle_FO.acquire() # Adiquirindo uma posição do semáforo. Caso não hajam posições disponíveis para
                                      # o algoritmo espera o Controle_FO.release
                fitness = FO(vetor_posicoes[ID_particle], self.args_model)
                fitness.start()
                threadarray.append(fitness)
                ID_particle += 1

        # Caso o método de busca seja Regiao, o ponto focal é forçadamente adicionado ao vetor_posicoes
        if self.metodo.busca == 'Regiao':
            vetor_posicoes[0] = self.foco

        # Inicializando os melhores valores das partículas (vetor_pbest) - variável global
        # neste ponto, pbest é o próprio vetor_posicoes e vetor_fitness
        vetor_pbest = [vetor_posicoes,vetor_fitness]

        # Inicializando melhores valores globais do enxame (gbest)
        best_ID_particle = argmin(vetor_fitness) # partícula com menor valor de função objetivo
        best_fitness = vetor_fitness[best_ID_particle] # valor da função objetivo no ponto ótimo - variável global
        if self.metodo.busca == 'Otimo':
            gbest = vetor_posicoes[best_ID_particle] # variável global
        elif self.metodo.busca == 'Regiao':
            gbest = self.foco

        # ----------------------------------------------------------------------------------------
        # ARMAZENAMENTO NO HISTÓRICO
        # ----------------------------------------------------------------------------------------
        self.historico_fitness[0] = copy(vetor_fitness).tolist()
        self.historico_posicoes[0] = copy(vetor_posicoes).tolist()
        self.media_fitness[0] = mean(vetor_fitness)
        self.desvio_fitness[0] = std(vetor_fitness, ddof=1)/max([best_fitness, 1e-17])
        self.historico_best_fitness[0] = best_fitness
        self.historico_w[0] = self.w[0]

        aux1 = []
        for j in xrange(self.Num_particulas):
            aux2 = [abs(vetor_velocidades[j][k]) for k in xrange(self.Num_parametros)]
            aux1.append(aux2)

        self.media_velocidade[0] = mean(aux1)
        self.velocidade_ideal[0] = self.Vstart

        # ----------------------------------------------------------------------------------------
        # DESENVOLVIMENTO DAS ITERAÇÕES
        # ----------------------------------------------------------------------------------------
        # lista com o número das partículas (evitar um range dentro do for) - obtem todos os ID's das partículas
        vetor_Num_particulas = range(self.Num_particulas)

        if self.index_desempenho[-1] != self.itmax-1: #forçando que o último valor salvo seja da última iteração
            self.index_desempenho[-1] = self.itmax-1
        # Inicialização de contadores
        ithist = 0       # contador para as listas de histórico histórico
        itdesempenho = 1 # contador para as listas de desemoenho

        # Inicialização de variáveis atributos do PSO
        velocidade_ideal = 0
        w = self.w[0]
        C1 = None
        C2 = None
        Vreinit = None

        # Desenvolvimento das iterações, a partir da 1, pois a 0 é a inicialização
        for it in xrange(1, self.itmax):

            total_particulas_atendidas = 0  # Contagem de partículas que finalizaram a execução

            # Aquisição da Lock Controle_Iteração, para iniciar a iteração, o release só ocorre quando todas as partículas
            # terminarem sua execução
            Controle_Iteracao.acquire()

            # Caso a keyword printit seja True, será printado
            if printit == True:
                sys.stdout.write('ITERACAO: ' + str(it) + '\n')
                sys.stdout.flush()

            # Calculando média das velocidades:
            vel_aux = []
            for j in xrange(self.Num_particulas):
                vel_aux.append([abs(vetor_velocidades[j][k]) for k in xrange(self.Num_parametros)])

            media_velocidade = mean(vel_aux)

            # Atualização de w, C1, C2 e Vreinit
            if self.metodo.inercia == 'Constante':
                w = self.w[0]
            elif self.metodo.inercia == 'TVIW-linear':
                w = self.w[0] + (float(it) - 1) / (1 - float(self.itmax - 1)) * (self.w[0] - self.w[1])
            elif self.metodo.inercia == 'TVIW-random':
                w = 0.5 + random.uniform(0.0, 1.0) / 2
            elif self.metodo.inercia == 'TVIW-Adaptative-VI':

                velocidade_ideal = self.Vstart * (1 + cos(it * pi / self.Tend)) / 2

                if media_velocidade >= velocidade_ideal:
                    w = max([w - self.deltaw, self.w[1]])
                else:
                    w = min([w + self.deltaw, self.w[0]])

            elif self.metodo.inercia is None:
                w = None

            if self.metodo.aceleracao == 'Constante':
                C1 = self.C1[0]
                C2 = self.C2[0]
            elif self.metodo.aceleracao == 'TVAC':
                C1 = self.C1[0] + (float(it) - 1) / (1 - float(self.itmax - 1)) * (self.C1[0] - self.C1[1])
                C2 = self.C2[0] + (float(it) - 1) / (1 - float(self.itmax - 1)) * (self.C2[0] - self.C2[1])

            if self.metodo.algoritmo == 'HPSO':
                Vreinit = []
                for i in xrange(self.Num_parametros):
                    Vreinit.append(self.Vreinit[0][i] + (float(it) - 1) / (1 - float(self.itmax - 1)) * (
                        self.Vreinit[0][i] - self.Vreinit[1] * self.Vreinit[0][i]))

            # Create new Threads (Partículas)
            for ID_particle in vetor_Num_particulas:
                Controle_Particula.acquire()  # Aquisião do Semáforo -> limitar a quantidade de particulas sendo executadas ao mesmo tempo
                particula = Particula(FO, w, C1, C2, self.Vmax, Vreinit,
                                      self.Num_parametros, self.metodo, self.limite_superior,
                                      self.limite_inferior, ID_particle, self.Num_particulas, self.args_model)
                particula.start()

            # A execução aguarda a liberação (release) de Controle_Iteração, pois daqui em diantes todas as partículas
            # devem ter finalizado
            Controle_Iteracao.acquire()

            # Caso o método gbest seja Enxame ele é aqui atualizado e não dentro das partículas
            if self.metodo.gbest == 'Enxame':
                if min(vetor_fitness) < best_fitness:
                    best_ID_particle = argmin(vetor_fitness)
                    best_fitness = vetor_fitness[best_ID_particle]
                    gbest = vetor_posicoes[best_ID_particle]

            # Armazenamento de informações
            # Armazenamento do histótico das posicoe e fitness
            if it >= self.itmax - self.n_historico: # só será armazenado para as n_historico últimas iterações

                self.historico_fitness[ithist] = copy(vetor_fitness).tolist()
                self.historico_posicoes[ithist] = copy(vetor_posicoes).tolist()
                ithist+=1

            # Desempenho
            if self.index_desempenho[itdesempenho] == it: # desempenho só é salvo em pontos específicos

                self.media_fitness[itdesempenho] = mean(vetor_fitness)
                self.desvio_fitness[itdesempenho] = std(vetor_fitness, ddof=1)/max([best_fitness, 1e-17])
                self.historico_w[itdesempenho] = w
                self.historico_best_fitness[itdesempenho] = best_fitness
                self.velocidade_ideal[itdesempenho] = velocidade_ideal

                aux1 = []
                for j in xrange(self.Num_particulas):
                    aux2 = [abs(vetor_velocidades[j][k]) for k in xrange(self.Num_parametros)]
                    aux1.append(aux2)

                self.media_velocidade[itdesempenho] = mean(aux1)

                itdesempenho+=1

            # Liberação da proxima iteração
            Controle_Iteracao.release()

        # ----------------------------------------------------------------------------------------
        # PROCEDIMENTOS FINAIS
        # ----------------------------------------------------------------------------------------
        # Armazenar gbest e best_fitness
        self.gbest = gbest
        self.best_fitness = best_fitness

        for ithist in xrange(self.n_historico):
            for ID_particula in xrange(self.Num_particulas):
                if isnan(self.historico_fitness[ithist][ID_particula]):
                    raise ValueError('Existe NaN como valor de função objetivo. Verificar.')

    def Relatorios(self, base_path=None):
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
            base_path = os.getcwd() + "/PSO/Result_txt/"
        Validacao_Diretorio(base_path)

        savetxt(base_path + 'Best_fitness.txt', matrix(self.best_fitness), fmt='%.18f')
        outfile = open(base_path + 'historico_fitness.txt', 'w')
        for elemento in self.historico_fitness:
            outfile.write(str(elemento))
            outfile.write('\n')
        outfile.close()

        outfile = open(base_path + 'historico_posicoes.txt', 'w')
        for elemento in self.historico_posicoes:
            outfile.write(str(elemento))
            outfile.write('\n')
        outfile.close()

        savetxt(base_path + 'gbest.txt', matrix(self.gbest), fmt='%.18f')

        # Resumo da estimação
        outfile = codecs.open(base_path + 'Resumo.txt', 'w', 'utf-8')
        outfile.write(u'RESUMO DO PSO\n')
        outfile.write('--------------------------------' + '\n')
        outfile.write(u'MÉTODO' + '\n')
        outfile.write(u'Algoritmo: %s | Inercia: %s | Aceleracão: %s | Vreinit: %s' % (self.metodo.algoritmo,
                                                                                       self.metodo.inercia,
                                                                                       self.metodo.aceleracao,
                                                                                       self.metodo.Vreinit) + '\n')
        outfile.write('--------------------------------' + '\n')
        outfile.write(u'Seleção dos parâmetros, partículas e iterações' + '\n')
        outfile.write(u'INÍCIO - w: %s | c1: %s | c2: %s' % (self.w[0], self.C1[0], self.C2[0]) + '\n')
        outfile.write(u'FIM    - w: %s | c1: %s | c2: %s' % (self.w[1], self.C1[1], self.C2[1]) + '\n')
        outfile.write(u'Vmax: %s' % self.Vmax + '\n')
        outfile.write(u'Vreinit: %s' % self.Vreinit + '\n')
        outfile.write(u'deltaw: %s' % self.deltaw + '\n')
        outfile.write(u'Número de partículas: %s' % (self.Num_particulas,) + '\n')
        outfile.write(u'Iterações: %s (1 de inicialização, %s de Busca)' % (self.itmax, self.itmax - 1) + '\n')
        outfile.write('--------------------------------' + '\n')
        outfile.write(u'RESULTADOS' + '\n')
        outfile.write(u'ótimo: %s | fitness: %s \n' % (self.gbest, self.best_fitness))
        outfile.write(u'Desvio relativo: %s\n'%(self.desvio_fitness[self.n_desempenho-1]/max([1e-16,self.best_fitness])))
        outfile.write(u'Desvio: %s\n'%(self.desvio_fitness[self.n_desempenho-1]))
        outfile.write(u'Fitness: %s\n'%(self.best_fitness))
        outfile.close()

    def Graficos(self, base_path=None, **kwargs):
        """
        Método para criação dos gráficos de indicadores do PSO, incluindo a função objetivo.

        =========
        Entradas
        =========

        * ``base_path`` (string, opcional): caminho completo onde os gráficos serão criados.


        ======
        Saídas
        ======
        Gráficos:
        - Baseados no histórico: (O tamanho do histórico afeta estes gráficos)
        * Funcao objetivo em três dimensões:
        * Função objetivo em 1 dimensão
        * Histograma dos parâmetros

        - Baseados no Desempenho:
        * Media e desvio padrão do fitness das partículas
        * fator de inércia
        * velocidade

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

        """
        # kwargs: legenda_posicao, azim, elev, Nome_param, Unid_param
        # Atribuição de valores default
        legenda_posicao = 1
        azim = [45]  # Define a rotação horizontal no gráfico da função objetivo
        elev = [45]  # Define a rotação vertical no gráfico da função objetivo
        Nome_param = []
        Unid_param = []
        FO2a2 = False
        # Posições de Legenda:
        # Superior Direita  = 1
        # Superior Esquerda = 2
        # Inferior Esquerda = 3
        # Inferior Direita  = 4
        # Direita           = 5
        # Centro Esquerda   = 6
        # Centro Direita    = 7
        # Centro inferior   = 8
        # Centro Superior   = 9
        # Centro            = 10

        # Sobreescrevendo valores default
        if 'legenda_posicao' in kwargs.keys():
            legenda_posicao = kwargs['legenda_posicao']

        if 'azim' in kwargs.keys():
            azim = kwargs['azim']

        if 'elev' in kwargs.keys():
            elev = kwargs['elev']

        if 'Nome_param' in kwargs.keys():
            Nome_param = kwargs['Nome_param']
            if Nome_param is not None:
                if len(Nome_param) != self.Num_parametros and len(Nome_param) != 0:
                    raise NameError, u'Nome_param deve conter o nome para todos os parâmetros'

        if 'Unid_param' in kwargs.keys():
            Unid_param = kwargs['Unid_param']
            if Unid_param is not None:
                if len(Unid_param) != self.Num_parametros and len(Unid_param) != 0:
                    raise NameError, u'Unid_param deve conter as unidades para todos os parâmetros'

            if all([unid is None for unid in Unid_param]):
                Unid_param = None

        if 'FO2a2' in kwargs.keys():
            if kwargs['FO2a2'] == False:
                FO2a2 = False
            else:
                FO2a2 = True

        if isinstance(azim, int) or isinstance(azim, float) or isinstance(elev, int) or isinstance(elev, float):
            raise NameError, u'keywords azim e elev devem ser listas'

        # Verificação da existência de diretório do diretório
        if base_path is None:
            base_path = os.getcwd() + "/PSO/Graficos/"
        Validacao_Diretorio(base_path)

        # Gráficos de Desempenho do algoritmo de PSO-----------------------------------------------

        # Uso do formato científico nos gráficos:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        # Desenvolvimento das iterações - Fitness e desvio do fitness
        fig = figure()
        ax = fig.add_subplot(2, 1, 1)
        plot(self.index_desempenho, self.media_fitness, 'b-', label=u'Média')
        plot(self.index_desempenho, self.historico_best_fitness, 'r-', label=u'Best Fitness')
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        xlim((0, self.itmax))
        ylabel(u"Média de " + r"$\Phi$", fontsize=15)
        legend(loc=legenda_posicao)

        ax2 = fig.add_subplot(2, 1, 2)
        plot(self.index_desempenho, self.desvio_fitness, 'b-', label=u'Desvio-padrão')
        ax2.yaxis.grid(color='gray', linestyle='dashed')
        ax2.xaxis.grid(color='gray', linestyle='dashed')
        xlim((0, self.itmax))
        xlabel(u"Iterações", fontsize=15)
        ylabel(u"Desvio-padrão relativo de " + r"$\Phi$", fontsize=15)
        legend(loc=legenda_posicao)
        ax.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path + 'Medias_Fitness_global.png')
        close()

        # Desenvolvimento das Iterações - Velocidades
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        plot(self.index_desempenho, self.media_velocidade, 'b-')
        if self.metodo.inercia == 'TVIW-Adaptative-VI':
            plot(self.index_desempenho, self.velocidade_ideal, 'r-')
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        xlim((0, self.n_historico))
        xlabel(u"Iterações", fontsize=15)
        ylabel(u"Média de " + r"$\nu$", fontsize=15)
        ax.yaxis.set_major_formatter(formatter)
        fig.savefig(base_path + 'Medias_velocidade_global.png')
        close()

        if (self.metodo.inercia == 'TVIW-linear') or (self.metodo.inercia == 'TVIW-random') or (self.metodo.inercia == 'TVIW-Adaptative-VI'):
            # Gráfico de histório do inertia weight
            fig = figure()
            ax3 = fig.add_subplot(1, 1, 1)
            plot(self.index_desempenho, self.historico_w, 'b-')
            ax3.yaxis.grid(color='gray', linestyle='dashed')
            ax3.xaxis.grid(color='gray', linestyle='dashed')
            xlim((0, self.itmax))
            if self.metodo.inercia == 'TVIW-linear':
                ylim((self.w[1], self.w[0]))
            xlabel(u"Iterações")
            ylabel(u"w")
            ax3.yaxis.set_major_formatter(formatter)
            fig.savefig(base_path + 'W_inertia_weight.png')
            close()

        # Transformação do histórico das posições e do fitness
        hist_posicoes = []
        hist_fitness = []
        for it in xrange(self.n_historico):
            for ID_particula in xrange(self.Num_particulas):
                hist_posicoes.append(self.historico_posicoes[it][ID_particula])
                hist_fitness.append(self.historico_fitness[it][ID_particula])

        # Histograma das posições
        for D in xrange(self.Num_parametros):
            aux = [hist_posicoes[it][D] for it in xrange(self.n_historico)]  # obtenção das posições para a dimensão D

            fig = figure()
            ax = fig.add_subplot(1, 1, 1)
            hist(aux, normed=True, histtype='stepfilled', color='b')
            ylabel(u'Frequência normalizada')
            xlabel(u'Valores dos parâmetros')
            fig.savefig(base_path + 'Histograma_parametros_%d' % (D + 1) + '.png')

        # Gráfico da função objetivo----------------------------------------------
        for ix in xrange(self.Num_parametros):
            # Gráfico unidimensional da função objetivo
            # Desenvolvimento das Iterações - Velocidades
            X = [hist_posicoes[it][ix] for it in xrange(len(hist_posicoes))]
            Y = hist_fitness

            X_sort = sort(X)
            Y_sort = [Y[i] for i in argsort(X)]

            fig = figure()
            ax = fig.add_subplot(1, 1, 1)
            plot(X_sort, Y_sort, 'bo', markersize=4)
            plot(self.gbest[ix], self.best_fitness, 'ro', markersize=8)
            ax.yaxis.grid(color='gray', linestyle='dashed')
            ax.xaxis.grid(color='gray', linestyle='dashed')
            ylabel(r"$\quad \Phi $", fontsize=20)

            if isinstance(Nome_param, list):
                if len(Nome_param) == 0:
                    xlabel(r"$x_{%d}$ " % (ix), fontsize=20)
                elif Unid_param is None:
                    xlabel(Nome_param[ix], fontsize=20)
                elif isinstance(Unid_param, list):
                    if len(Unid_param) == 0:
                        xlabel(Nome_param[ix], fontsize=20)
                    else:
                        xlabel(Nome_param[ix] + u'/' + Unid_param[ix], fontsize=20)
            else:
                xlabel(r"$x_{%d}$ " % (ix + 1), fontsize=20)
            ax.yaxis.set_major_formatter(formatter)
            fig.savefig(base_path + 'Funcao_objetivo_projecao_' + 'x%d' % (ix + 1) + '.png')
            close()

        if (self.Num_parametros != 1) and (FO2a2 == True):
            # Gráfico tridimensioal da função objetivo
            Combinacoes = int(factorial(self.Num_parametros) / (factorial(self.Num_parametros - 2) * factorial(2)))
            p1 = 0
            p2 = 1
            cont = 0
            passo = 1

            for pos in xrange(Combinacoes):
                if pos == (self.Num_parametros - 1) + cont:
                    p1 += 1
                    p2 = p1 + 1
                    passo += 1
                    cont += self.Num_parametros - passo
                X = [hist_posicoes[it][p1] for it in xrange(len(hist_posicoes))]
                Y = [hist_posicoes[it][p2] for it in xrange(len(hist_posicoes))]
                Z = hist_fitness

                Z_sort = sort(Z)
                Y_sort = [Y[i] for i in argsort(Z)]
                X_sort = [X[i] for i in argsort(Z)]

                fig = figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_sort, Y_sort, Z_sort, c=Z_sort, cmap=cm.coolwarm, zorder=1)
                arrow = Arrow3D([self.gbest[p1], self.gbest[p1]], [self.gbest[p2], self.gbest[p2]],
                                [Z_sort[-1] / 4, Z_sort[0]], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
                ax.add_artist(arrow)
                ax.yaxis.grid(color='gray', linestyle='dashed')
                ax.xaxis.grid(color='gray', linestyle='dashed')
                ax.set_zlabel(r"$\quad \Phi $ ", rotation=180, fontsize=20)

                if isinstance(Nome_param, list):
                    if len(Nome_param) == 0:
                        ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                        ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)
                    elif Unid_param is None:
                        ax.set_ylabel(Nome_param[p2], fontsize=20)
                        ax.set_xlabel(Nome_param[p1], fontsize=20)
                    elif isinstance(Unid_param, list):
                        if len(Unid_param) == 0:
                            ax.set_ylabel(Nome_param[p2], fontsize=20)
                            ax.set_xlabel(Nome_param[p1], fontsize=20)
                        else:
                            ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize=20)
                            ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize=20)
                else:
                    ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                    ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)

                if len(azim) == 1:
                    ax.azim = azim[0]
                else:
                    ax.azim = azim[pos]

                if len(elev) == 1:
                    ax.elev = elev[0]
                else:
                    ax.elev = elev[pos]
                ax.set_zlim([Z_sort[0], Z_sort[-1]])
                ax.w_zaxis.set_major_formatter(formatter)
                fig.savefig(base_path + 'Funcao_objetivo' + '_x' + str(p1 + 1) + '_x' + str(p2 + 1) + '.png')
                close()
                p2 += 1

    def Movie(self, base_path=None, tipos=['evolucao', 'projecao', 'funcao'], **kwargs):
        """
        Método para criação dos gráficos de desenvolvimento do algoritmo a cada iteração. Cada gráfico representa um frame
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
        """
        # Atribuição de valores default
        azim = [45]  # Define a rotação horizontal no gráfico da função objetivo
        elev = [45]  # Define a rotação vertical no gráfico da função objetivo
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
                if len(Nome_param) != self.Num_parametros and len(Nome_param) != 0:
                    raise NameError, u'Nome_param deve conter o nome para todos os parâmetros'

        if 'Unid_param' in kwargs.keys():
            Unid_param = kwargs['Unid_param']
            if Unid_param != None:
                if len(Unid_param) != self.Num_parametros and len(Unid_param) != 0:
                    raise NameError, u'Unid_param deve conter as unidades para todos os parâmetros'

        if isinstance(azim, int) or isinstance(azim, float) or isinstance(elev, int) or isinstance(elev, float):
            raise NameError, u'keywords azim e elev devem ser listas'

        len_itmax = len(str(self.itmax))
        # Verificação da existência de diretório do diretório
        if base_path == None:
            base_path = os.getcwd() + "/PSO/"
        dir1 = base_path + "/Evolucao/"
        Validacao_Diretorio(dir1)

        # Formatação dos gráficos
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))

        if 'evolucao' in tipos:
            # Desenvolvimento das iterações:
            for i in xrange(self.itmax):
                fig = figure()
                ax = fig.add_subplot(111, projection='3d')
                xs = [i] * self.Num_particulas
                ys = range(0, self.Num_particulas)
                zs = self.historico_fitness[i]
                ax.scatter(xs, ys, zs, c='b', marker='o')
                ax.set_xlabel(u'Iteração')
                ax.set_ylabel(u'ID_Partícula')
                ax.set_zlabel(u'Fitness / u.m')
                xlim((0.0, self.itmax))
                len_it = len(str(i))
                numeracao = '0' * (len_itmax - len_it) + str(i)
                fig.savefig(dir1 + 'Evolucao_algoritmo_' + numeracao + '.png')
                close()

        if self.Num_parametros >= 2 and (('projecao' in tipos) or ('funcao' in tipos)):

            Combinacoes = int(factorial(self.Num_parametros) / (factorial(self.Num_parametros - 2) * factorial(2)))

            for it in xrange(self.itmax):
                p1 = 0;
                p2 = 1;
                cont = 0;
                passo = 1
                for pos in xrange(Combinacoes):
                    dir2 = base_path + "/Projecao_Combinacoes_" + str(pos) + "/"
                    Validacao_Diretorio(dir2)
                    dir3 = base_path + "/Funcao_Combinacoes_" + str(pos) + "/"
                    Validacao_Diretorio(dir3)
                    if pos == (self.Num_parametros - 1) + cont:
                        p1 += 1
                        p2 = p1 + 1
                        passo += 1
                        cont += self.Num_parametros - passo

                    len_it = len(str(it))
                    numeracao = '0' * (len_itmax - len_it) + str(it)

                    if 'projecao' in tipos:
                        fig = figure()
                        ax = fig.add_subplot(1, 1, 1)

                        for ID in xrange(self.Num_particulas):
                            plot(self.historico_posicoes[it][ID][p1], self.historico_posicoes[it][ID][p2], 'bo',
                                 linewidth=2.0)

                        plot(self.gbest[p1], self.gbest[p2], 'r*', markersize=10.0)
                        ax.yaxis.grid(color='gray', linestyle='dashed')
                        ax.xaxis.grid(color='gray', linestyle='dashed')

                        if isinstance(Nome_param, list):
                            if len(Nome_param) == 0:
                                ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                                ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)
                            elif isinstance(Unid_param, list):
                                if len(Unid_param) == 0:
                                    ax.set_ylabel(Nome_param[p2], fontsize=20)
                                    ax.set_xlabel(Nome_param[p1], fontsize=20)
                            elif Unid_param == None:
                                ax.set_ylabel(Nome_param[p2], fontsize=20)
                                ax.set_xlabel(Nome_param[p1], fontsize=20)
                            else:
                                ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize=20)
                                ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize=20)
                        else:
                            ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                            ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)

                        xlim((self.limite_inferior[p1], self.limite_superior[p1]))
                        ylim((self.limite_inferior[p2], self.limite_superior[p2]))

                        fig.savefig(
                            dir2 + 'Parametros' + '_p' + str(p1 + 1) + '_p' + str(p2 + 1) + '_it' + numeracao + '.png')
                        close()

                    if 'funcao' in tipos:
                        hist_posicoes = [];
                        hist_fitness = []
                        for ID_particula in xrange(self.Num_particulas):
                            hist_posicoes.append(self.historico_posicoes[it][ID_particula])
                            hist_fitness.append(self.historico_fitness[it][ID_particula])

                        X = [hist_posicoes[it][p1] for it in xrange(len(hist_posicoes))]
                        Y = [hist_posicoes[it][p2] for it in xrange(len(hist_posicoes))]
                        Z = hist_fitness

                        Z_sort = sort(Z)
                        Y_sort = [Y[i] for i in argsort(Z)]
                        X_sort = [X[i] for i in argsort(Z)]

                        fig = figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(X_sort, Y_sort, Z_sort, c=Z_sort, cmap=cm.coolwarm, zorder=1)
                        arrow = Arrow3D([self.gbest[p1], self.gbest[p1]], [self.gbest[p2], self.gbest[p2]],
                                        [max(self.historico_fitness) / 4, min(self.historico_fitness)],
                                        mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
                        ax.add_artist(arrow)
                        ax.yaxis.grid(color='gray', linestyle='dashed')
                        ax.xaxis.grid(color='gray', linestyle='dashed')
                        ax.set_zlabel(r"$\quad \Phi $ ", rotation=180, fontsize=20)

                        ax.set_zlim([min(self.historico_fitness), max(self.historico_fitness)])
                        ax.set_ylim([self.limite_inferior[p2], self.limite_superior[p2]])
                        ax.set_xlim([self.limite_inferior[p1], self.limite_superior[p1]])

                        if isinstance(Nome_param, list):
                            if len(Nome_param) == 0:
                                ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                                ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)
                            elif isinstance(Unid_param, list):
                                if len(Unid_param) == 0:
                                    ax.set_ylabel(Nome_param[p2], fontsize=20)
                                    ax.set_xlabel(Nome_param[p1], fontsize=20)
                            elif Unid_param == None:
                                ax.set_ylabel(Nome_param[p2], fontsize=20)
                                ax.set_xlabel(Nome_param[p1], fontsize=20)
                            else:
                                ax.set_ylabel(Nome_param[p2] + u'/' + Unid_param[p2], fontsize=20)
                                ax.set_xlabel(Nome_param[p1] + u'/' + Unid_param[p1], fontsize=20)
                        else:
                            ax.set_ylabel(r"$x_{%0.0f}$ " % (p2 + 1), fontsize=20)
                            ax.set_xlabel(r"$x_{%0.0f}$ " % (p1 + 1), fontsize=20)

                        if len(azim) == 1:
                            ax.azim = azim[0]
                        else:
                            ax.azim = azim[pos]

                        if len(elev) == 1:
                            ax.elev = elev[0]
                        else:
                            ax.elev = elev[pos]
                        ax.w_zaxis.set_major_formatter(formatter)
                        fig.savefig(dir3 + 'Funcao_objetivo' + '_x' + str(p1 + 1) + '_x' + str(
                            p2 + 1) + '_it' + numeracao + '.png')
                        close()

                    p2 += 1

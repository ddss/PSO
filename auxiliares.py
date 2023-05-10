# -*- coding: utf-8 -*-

'''
Arquivo contendo subrotinas auxiliares para a classe PSO
'''

import os
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

# -------------------------------------------------------------------------------
# Subrotinas e classes auxiliares
# -------------------------------------------------------------------------------
def Validacao_Diretorio(diretorio):
    # Basedo em códigos de terceiros (Based on interne:
    # Validação da existência de diretório
    directory = os.path.split(diretorio + "Teste.txt")[0]

    if directory == '':
        directory = '.'

    # Se o diretório não existir, crie
    if not os.path.exists(directory):
        os.makedirs(directory)


class Arrow3D(FancyArrowPatch):
    # Seta em gráfico 3D
    # http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
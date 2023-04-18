"""
Graphics
@author: Renilton
"""
from numpy import array, linspace, shape, sqrt, average, append, vstack, std, zeros
import plotly.graph_objects as go
from bokeh.plotting import figure, show
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches


class Graph:
    def __init__(self, coordinates, fit, vel, bounds, optimal_point, optimal_fit, inter, num_part):
        """
        Class used to define the particles
        Attributes
        ----------
        coordinates : array
            a list with the coordinates of each particle in each interaction
        fit : array
            a list with the fitness of each particle in each interaction
        vel : array
            a list with the velocity of each particle in each interaction
        vel_ave : array
            a list with the velocity average of each particle in each interaction
        fit_ave: array
            a list with the fitness average of each particle in each interaction
        vel_desv: array
            a list with the standart desviation of velocity average
        fit_desv: array
            a list with the standart desviation of fitness average
        bounds: array with shape (nd x 2)
            position restriction
        optimal_point: float
            the optimal point
        optimal_fit: float
            the fitness of the optimal point
        """
        self.posit = coordinates
        self.fit = fit
        self.vel = vel
        self.dim = bounds.shape[0]
        self.optimal_point = optimal_point
        self.optimal_fit = optimal_fit
        self.inter = inter
        self.num_part = num_part


    def pos_fit_3d(self):
        label = ['x{}'.format(i+1) for i in range(self.dim)]
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                if j >= i:
                    fig = go.Figure(data=[go.Scatter3d(
                        x=self.posit[i],
                        y=self.posit[j+1],
                        z=self.fit,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=self.fit,  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.8
                        )
                    )])
                    fig.update_layout(scene=dict(
                                        xaxis=dict(title=(label[i])),
                                        yaxis=dict(title=(label[j+1])),
                                        zaxis=dict(title='Fit')), #, range=(0,40))),
                                        title='Fit versus Positions(x,y)',
                                        margin=dict(r=0, l=0, b=0, t=30)
                                      )
                    fig.show()



    # Positions
    def positions(self):
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                if j >= i:
                    mp.figure()
                    mp.title('Positions x[{}] vs x[{}]'.format(i+1,j+2))
                    mp.xlabel('x[{}]'.format(i+1))
                    mp.ylabel('x[{}]'.format(j+2))
                    levels = linspace(self.fit.min(), self.fit.max(), 10)
                    mp.tricontourf(self.posit[i], self.posit[(j + 1)], self.fit, levels=levels, alpha=0.7)
                    mp.colorbar()
                    mp.plot(self.posit[i], self.posit[(j + 1)], '.', label='x[{}] vs x[{}]'.format(i+1,j+2), color='#202020')
                    mp.plot(self.optimal_point[i], self.optimal_point[j+i], '.', color='red', label='optimal point')
                    mp.legend()
                    mp.show()

    # Positions X Fitness
    def pos_fit_2d(self):
        for i in range(self.dim):
            mp.figure()
            mp.title('Position x[{}] vs Fit'.format(i+1))
            mp.xlabel('x[{}]'.format(i+1))
            mp.ylabel("fit")
            mp.plot(self.posit[i], self.fit, '.', label='x[{}] vs fit'.format(i+1))
            mp.plot(self.optimal_point[i], self.optimal_fit, '.', color='red', label='optimal point')
            mp.legend()
            mp.show()

    # Interactions(x) X Fitness(y)
    def int_fitness(self):
        mp.figure()
        mp.title('Interactions vs Fit')
        mp.xlabel("Interactions")
        mp.ylabel("Fit")
        mp.plot(self.fit, '.', label='interaction vs fitness')
        mp.legend()
        mp.show()

    # Interactions(x) X Fitness average(y)
    def int_fit_average(self):
        mp.figure()
        mp.title('Interactions vs Fitness average')
        mp.xlabel("Interactions")
        mp.ylabel("Fitness average")
        self.ave = array([])
        for i in range(0, self.inter):
            self.ave = append(self.ave, average(self.fit[(self.num_part*i):(self.num_part*(i+1))]))
        mp.plot(self.ave, '-', label='fitness average')
        mp.legend()
        mp.show()

    # Interactions(x) X Fitness standart desviation(y)
    def int_fit_sd(self):
        mp.figure()
        mp.title('Interactions vs Fitness standart desviation')
        mp.xlabel("Interactions")
        mp.ylabel("Fitness standart desviation")
        sd = array([])
        aux = zeros(self.num_part)
        for i in range(0, self.inter):
            aux[:] = self.ave[i]
            # print((self.fit[(self.num_part*i):(self.num_part*(i+1))]))
            # print(aux)
            sd = append(sd, std((self.fit[(self.num_part*i):(self.num_part*(i+1))], aux)))
        mp.plot(sd, '-', label='standart desviation')
        mp.legend()
        mp.show()

    # Interactions(x) X Velocity(x)
    def int_velocity(self):
        mp.figure()
        mp.title('Interactions vs Velocity')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        for i in range(self.dim):
            mp.plot(self.vel[i], '.', label='v[{}]'.format(i+1), alpha=(1/(i+1)))
        mp.legend()
        mp.show()

    # Interactions(x) X Velocity Average(x)
    def int_vel_average(self):
        mp.figure()
        mp.title('Interactions vs Velocity Average')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity average")
        avev = [[0], [0], [0]]
        for i in range(0, self.dim):
            for j in range(0, (shape(self.vel)[1])):
                avev[i] = vstack((avev[i], average(self.vel[i, :j+1])))
        for i in range(self.dim):
            mp.plot(avev[i], '-', label='v[{}]'.format(i+1))
        mp.legend()
        mp.show()

    # Interactions(x) X Velocity standart desviation(x)
    def int_vel_sd(self):
        mp.figure()
        mp.title('Interactions vs Velocity standart desviation')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity standart desviation")
        for i in range(self.dim):
            mp.plot(self.vel_desv[i], '-', label='v[{}] standart desviation'.format(i+1))
        mp.legend()
        mp.show()

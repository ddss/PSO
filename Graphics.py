"""
Graphics
@author: Renilton
"""
from numpy import array, linspace, shape, sqrt
import plotly.graph_objects as go
from bokeh.plotting import figure, show
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches


class Graph:
    def __init__(self, coordinates, fit, vel, vel_ave, fit_ave, vel_desv, fit_desv, bounds, optimal_point, optimal_fit):
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
        """
        self.posit = coordinates
        self.fit = fit
        self.vel = vel
        self.vel_ave = vel_ave
        self.fit_ave = fit_ave
        self.vel_desv = vel_desv
        self.fit_desv = fit_desv
        self.dim = bounds.shape[0]
        self.optimal_point = optimal_point
        self.optimal_fit = optimal_fit


    # def bokeh(self):
    #     radii = 0.01
    #     colors = array([[r, g, 150] for r, g in zip(50+100*abs(self.x), 50+100*abs(self.y))], dtype="uint8")
    #     tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,help"
    #     p = figure(title="Positions (x,y)", tools=tools)
    #     p.scatter(self.x, self.y, radius=radii, fill_color=colors, fill_alpha=1, line_color=None, name="PSO")
    #     p.xaxis.axis_label = 'X'
    #     p.yaxis.axis_label = 'Y'
    #     show(p)
    def plotly(self):
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

    def matplotlib(self):
        # Positions
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                if j >= i:
                    mp.figure()
                    mp.title('Positions x[{}] vs x[{}]'.format(i+1,j+2))
                    mp.xlabel('x[{}]'.format(i+1))
                    mp.ylabel('x[{}]'.format(j+2))
                    levels = linspace(self.fit.min(), self.fit.max(), 10)
                    mp.tricontourf(self.posit[i], self.posit[(j + 1)], self.fit, levels=levels, alpha=0.7)
                    mp.plot(self.posit[i], self.posit[(j + 1)], '.', label='x[{}] vs x[{}]'.format(i+1,j+2), color='#202020')
                    mp.plot(self.optimal_point[i], self.optimal_point[j+i], '.', color='red', label='optimal point')
                    mp.legend()
                    mp.show()
        # Positions X Fitness
        for i in range(self.dim):
            mp.figure()
            mp.title('Position x[{}] vs Fit'.format(i+1))
            mp.xlabel('x[{}]'.format(i+1))
            mp.ylabel("fit")
            mp.plot(self.posit[i], self.fit, '.', label='x[{}] vs fit'.format(i+1))
            mp.plot(self.optimal_point[i], self.optimal_fit, '.', color='red', label='optimal point')
            mp.legend()
            mp.show()
        # Interactions(x) X Fitness(x)
        mp.figure()
        mp.title('Interactions vs Fit')
        mp.xlabel("Interactions")
        mp.ylabel("Fit")
        mp.plot(self.fit, '.', label='interaction vs fitness')
        mp.legend()
        mp.show()
        # Interactions(x) X Fitness average(x)
        mp.figure()
        mp.title('Interactions vs Fitness average')
        mp.xlabel("Interactions")
        mp.ylabel("Fitness average")
        mp.plot(self.fit_ave, '-', label='fitness average')
        mp.plot(self.fit_desv, '-', label='standart desviation')
        mp.legend()
        mp.show()
        # Interactions(x) X Velocity(x)
        mp.figure()
        mp.title('Interactions vs Velocity')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        for i in range(self.dim):
            mp.plot(self.vel[i], '.', label='x[{}]'.format(i+1), alpha=(1/(i+1)))
        mp.legend()
        mp.show()
        # Interactions(x) X Velocity Average(x)
        mp.figure()
        mp.title('Interactions vs Velocity Average')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        for i in range(self.dim):
            mp.plot(self.vel_ave[i], '-', label='x[{}]'.format(i+1))
            mp.plot(self.vel_desv[i], '-', label='x[{}] standart desviation'.format(i+1))
        mp.legend()
        mp.show()

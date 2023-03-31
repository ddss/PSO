"""
Graphics
@author: Renilton
"""
from numpy import array
import plotly.graph_objects as go
from bokeh.plotting import figure, show
import matplotlib.pyplot as mp


class Graph:
    def __init__(self, coordinates, fit, vel, vel_ave, dim):
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
        self.dim = dim

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
        label = ['$x_{}$'.format(i+1) for i in range(self.dim)]
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
                    fig.update_layout(scene=dict(zaxis=dict(title='Fit')),  #, range=(0,40))),
                                      title='Fit versus Positions(x,y)',
                                      margin=dict(r=0, l=0, b=0, t=30)
                                      )
                    fig.show()

    def matplotlib(self):
        # Positions
        label = ['$x_{}$'.format(i+1) for i in range(self.dim)]
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                if j >= i:
                    mp.figure()
                    #mp.title(} x "label[i])
                    mp.xlabel(label[i])
                    mp.ylabel(label[j+1])
                    mp.plot(self.posit[i], self.posit[(j + 1)], '.')
                    mp.show()
        # Positions X Fitness
        for i in range(self.dim):
            mp.figure()
            #mp.title("x[i] versus fit")
            mp.xlabel(label[i])
            mp.ylabel("fit")
            mp.plot(self.posit[i], self.fit, '.')
            mp.show()
        # Interactions(x) X Fitness(x)
        mp.figure()
        mp.title('Interactions versus Fit')
        mp.xlabel("Interactions")
        mp.ylabel("Fit")
        mp.plot(self.fit, '.')
        mp.show()
        # Interactions(x) X Velocity(x)
        mp.figure()
        mp.title('Interactions versus Velocity')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        mp.plot(self.vel[0], '.')
        mp.plot(self.vel[1], '.')
        mp.show()
        # Interactions(x) X Velocity Average(x)
        mp.figure()
        mp.title('Interactions versus Velocity Average')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        mp.plot(self.vel_ave[0], '.')
        mp.plot(self.vel_ave[1], '.')
        mp.show()

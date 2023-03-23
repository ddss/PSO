"""
Graphics
@author: Renilton
"""
from numpy import array
import plotly.graph_objects as go
from bokeh.plotting import figure, show
import matplotlib.pyplot as mp


class Graph:
    def __init__(self, coordinates, fit, vel, vel_ave):
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
        self.x = coordinates[:, 0]
        self.y = coordinates[:, 1]
        self.fit = fit
        self.vel = vel
        self.vel_ave = vel_ave

    def bokeh(self):
        radii = 0.01
        colors = array([[r, g, 150] for r, g in zip(50+100*abs(self.x), 50+100*abs(self.y))], dtype="uint8")
        tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,help"
        p = figure(title="Positions (x,y)", tools=tools)
        p.scatter(self.x, self.y, radius=radii, fill_color=colors, fill_alpha=1, line_color=None, name="PSO")
        p.xaxis.axis_label = 'X'
        p.yaxis.axis_label = 'Y'
        show(p)

    def plotly(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.fit,
            mode='markers',
            marker=dict(
                size=3,
                color=self.fit,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )])
        fig.update_layout(scene=dict(zaxis=dict(title='Fit')),
                          title='Fit versus Positions(x,y)',
                          margin=dict(r=0, l=0, b=0, t=30)
                          )
        fig.show()

    def matplotlib(self, number_dimentions):
        # Positions
        for i in range(number_dimentions - 1):
            for j in range(number_dimentions - 1):
                if j >= i:
                    mp.figure()
                    mp.title("x[i] versus x[j]")
                    mp.xlabel("x[i]")
                    mp.ylabel("x[j]")
                    mp.plot(self.posit[1:, i], self.posit[1:, (j + 1)], '.')
                    mp.show()
        # Positions X Fitness
        for i in range(number_dimentions):
            mp.figure()
            mp.title("x[i] versus fit")
            mp.xlabel("x[i]")
            mp.ylabel("fit")
            mp.plot(self.posit[1:, i], self.fit[1:], '.')
            mp.show()
        # Interactions(x) X Fitness(x)
        mp.figure()
        mp.title('Interactions versus Fit')
        mp.xlabel("Interactions")
        mp.ylabel("Fit")
        mp.plot(self.fit[:], '.')
        mp.show()
        # Interactions(x) X Velocity(x)
        mp.figure()
        mp.title('Interactions versus Velocity')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        mp.plot(self.vel[:], '.')
        mp.show()
        # Interactions(x) X Velocity Average(x)
        mp.figure()
        mp.title('Interactions versus Velocity Average')
        mp.xlabel("Interactions")
        mp.ylabel("Velocity")
        mp.plot(self.vel_ave[:], '.')
        mp.show()

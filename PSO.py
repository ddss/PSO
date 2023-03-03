"""
@title : Particle Swarm Optimization (PSO)
@author: Renilton Ribeiro Almeida
"""

from numpy import sin, cos, sqrt, exp, e, array, append, random, vstack, pi
import plotly.graph_objects as go
from bokeh.plotting import figure, show
import matplotlib.pyplot as mp
from scipy.stats import qmc
import plotly.io as pio

pio.renderers.default = 'browser'

def myfunc(x,func_i,num_dim):
    """
    Function chosen to be optimized (minimize)

    Parameters
        ----------
        x : char
            function variable
        function_index : int
            index of the function to choose
        number_dimensions : int
            number of dimensions of the function
    """
    #Exponencial
    if func_i==1:
        total = 0
        for i in range(num_dim):
            total += x[i] ** 2
        return total
    #Rastrigin
    elif func_i==2:
        total = 0
        for i in range(num_dim):
            total += x[i]**2-10*cos(2*3.14*x[i])+10
        return total
    #Shaffer's f6
    elif func_i==3:
        total = 0
        for i in range(num_dim):
            total += 0.5+((sin(sqrt(x[i]**2)))**2-0.5)/(1.0 + 0.001*(x[i]**2))**2
        return total
    #Ackley
    elif func_i==4:
        total = 0
        for i in range(num_dim):
            total += -20.0 * exp(-0.2 * sqrt((x[i]**2)/2)) - exp(cos(2*3.14*x[i]))/2 + 20 + e
        return total

#--- MAIN ---------------------------------------------------------------------+
class Particle:

    def __init__(self, kwarg, i):
        """
        Class used to define the particles

        Attributes
        ----------
        position : array
            particle position
        velocity : array
            particle velocity
        pbest : array
            the best individual position
        fit_best : int
            the lowest individual value
        fit : list
            individual value

        Methods
        -------

        evaluate(myfunc)
            Evaluate the objetive function and record its value
        spso(g_best)
            Update particle velocity adding velocity cognitive and velocity social to velocity
        pso_wl(g_best, maxiter, i)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
        pso_wr(self, g_best)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
        update_velocity(gbest, max_inter, i)
            Update particle velocity
        update_position(self)
            Update particle position based on new velocity updates
        """
        self.dim = kwarg["number_dimentions"]
        self.bounds = kwarg["bounds"]
        self.c1 = kwarg["c1"]
        self.c2 = kwarg["c2"]
        self.wi = kwarg["wi"]
        self.wf = kwarg["wf"]
        self.maxiter = kwarg["interactions"]
        self.pbest= array([[],[]])
        self.fit_best=10000
        self.fit = array([])
        sobol_velocity = ([qmc.Sobol(d=self.dim).random_base2(m=5)])
        sobol_position = ([qmc.Sobol(d=self.dim).random_base2(m=5)])
        self.velocity = array((sobol_velocity[0][i]*(self.bounds[0][1]-self.bounds[0][0]))+(self.bounds[0][0]))
        self.position = array((sobol_position[0][i]*(self.bounds[0][1]-self.bounds[0][0]))+(self.bounds[0][0]))

    def evaluate(self, myfunc):
        """
        Evaluate the objetive function and record its value

        Parameters
        ----------
        myfunc: result of the function at a given point
        """
        self.fit=myfunc(self.position)
        if self.fit < self.fit_best:
            self.pbest = self.position
            self.fit_best = self.fit

    def spso(self, g_best):
        """
        The SPSO (Standard Particle Swarm Optimizer) update particle velocity
        adding velocity cognitive and velocity social to velocity

        Parameters
        ----------
        g_best: float
            record the best value of all values
        return: array
            The updated velocity through the SPSO method
        """
        r1 = random.random()
        r2 = random.random()
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        self.velocity = self.velocity + vel_cognitive + vel_social

    def pso_wl(self, g_best, maxiter, i):
        """
        The PSO_WL (Linear Time-Varying inercia Weight Particle Swarm Optimizer)
        update particle velocity multiplying velocity by inertia
        and adding velocity cognitive and velocity social to it

        Parameters
        ----------
        g_best: float
            record the best value of all values
        maxiter: int
            the amount of interactions
        i: int
            index
        return: array
            The updated velocity through the PSO_WL method
        """
        r1 = random.random()
        r2 = random.random()
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        w = (self.wi - self.wf) * ((maxiter - i) / maxiter) + self.wf
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def pso_wr(self, g_best):
        """
        The PSO_WR (Random Time-Varying Inertia Weight Particle Swarm Optimizer)
        update particle velocity multiplying velocity by inertia
        and adding velocity cognitive and velocity social to it

        Parameters
        ----------
        g_best: float
            record the best value of all values
        return: array
            The updated velocity through the PSO method
        """
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()
        w = 0.5 + (r3 / 2)
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def pso_chongpeng(self, g_best, i):
        """
        The PSO_WR (Random Time-Varying Inertia Weight Particle Swarm Optimizer)
        update particle velocity multiplying velocity by inertia
        and adding velocity cognitive and velocity social to it

        Parameters
        ----------
        g_best: float
            record the best value of all values
        i: int
            index
        return: array
            The updated velocity through the PSO method
        """
        r1 = random.random()
        r2 = random.random()
        k1 = random.randint(1)
        k2 = 1
        w = self.wf + ((self.wf - self.wi)*((1-(i/self.maxiter)**k1)**k2))
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def update_position(self, swarm):
        """
        Update particle position

        return: array
            The updated position, result of adding the old position and the new velocity
        """
        self.swarm = swarm

        self.position = self.position + self.velocity
        if abs(self.swarm.position[0]) < self.bounds[0][1] and abs(self.swarm.position[1]) > self.bounds[0][1]:
            self.swarm.position[1] = 2 * (self.swarm.position[1] / abs(self.swarm.position[1]))
            self.swarm.velocity = self.swarm.velocity * -0.1
        elif abs(self.swarm.position[0]) > self.bounds[0][1] and abs(self.swarm.position[1]) < self.bounds[0][1]:
            self.swarm.position[0] = 2 * (self.swarm.position[0] / abs(self.swarm.position[0]))
            self.swarm.velocity = self.swarm.velocity * -0.1
        elif abs(self.swarm.position[0]) > self.bounds[0][1] and abs(self.swarm.position[1]) > self.bounds[0][1]:
            self.swarm.position[0] = 2 * (self.swarm.position[0] / abs(self.swarm.position[0]))
            self.swarm.position[1] = 2 * (self.swarm.position[1] / abs(self.swarm.position[1]))
            self.swarm.velocity = self.swarm.velocity * -0.1
        # RESTRIÇÕES!!!!
        #x = array([1., 2., 3., 4., 5., 6.])
        #xmax = array([0.5, 0.5, 0.5, 10, 10, 10])
        # teste = x > xmax
        #x[x > xmax] = xmax[where(teste)[0]]
        # v[where(teste)[0]] = -PARAM_RED_VEL*v[where(teste)[0]]

class PSO():

    class history():
       def __init__(self, dim):
        self.position = array(random.rand(1,dim))
        self.fitness = array(random.rand(1, 1))
        self.velocity = array(random.rand(1,dim))

       def add(self, position, fitness, velocity):
            self.position = vstack([self.position, position])
            self.fitness = vstack([self.fitness, fitness])
            self.velocity = vstack([self.velocity, velocity])

    def __init__(self, myfunc, kwarg):
        """
        Class that uses particles in an optimization loop

        Parameters
        ----------
        myfunc: result of the function at a given point

        Attributes
        ----------
        swarm_positions : array
            an array to record the positional coordinates of each particle in each interaction
        swarm_fitness : array
            an array to record the value of the function in each position
        fit_gbest : float
            best group value
        gbest : list
            best group position
        swarm : list
            establishes the swarm
        """
        self.dim = kwarg["number_dimentions"]
        self.num_part = kwarg["number_particles"]
        self.max_inter = kwarg["interactions"]
        self.pso_version = kwarg["pso_version"]
        self.bounds = kwarg["bounds"]
        self.history = self.history(self.dim)
        self.fit_gbest = 10000
        self.gbest = []
        self.swarm = array([])

        for i in range(0, self.num_part):
            self.swarm = append(self.swarm, Particle(kwarg, i))
        i = 0
        p = 0
        g = 0
        velocity_sum = 0
        while i < self.max_inter and p < 1e3: # stopping criteria
            for j in range(0, self.num_part):
                self.swarm[j].evaluate(myfunc)
                velocity_sum = velocity_sum + self.swarm[j].velocity
                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)

                # determines if the current particle is the best (globally)
                if abs(self.swarm[j].fit-self.fit_gbest)<1e-6:
                    p+=1

                if self.swarm[j].fit < self.fit_gbest:
                    self.gbest = list(self.swarm[j].position)
                    self.fit_gbest = float(self.swarm[j].fit)


                if self.pso_version == 1:
                    self.swarm[j].spso(self.gbest)
                elif self.pso_version == 2:
                    self.swarm[j].pso_wl(self.gbest, self.max_inter, i)
                elif self.pso_version == 3:
                    self.swarm[j].pso_wr(self.gbest)
                elif self.pso_version == 4:
                    self.swarm[j].pso_chongpeng(self.gbest, i)
                self.swarm[j].update_position(self.swarm[j])
            g += 1
            i += 1
        self.velocity_average = (velocity_sum/g)*(1/(pi*(1+(random.uniform(self.bounds[0][0], self.bounds[0][1]))**2)))
        self.gbest_mo = self.gbest + self.velocity_average
        # ----------------------------------------------------------------------------------------
class Graph:
    def __init__(self, coordinates):

        """
        Class used to define the particles

        Attributes
        ----------
        x : array
            a list with the abscissa of each particle in each interaction
        y : array
            a list with the ordinate of each particle in each interaction
        radii : float
            the radius of the circle of the particles
        TOOLS : strig
            functions requested from the Bokeh
        p :
            the graphic
        """
        x = coordinates[:,0]
        y = coordinates[:,1]
        radii = 0.01
        colors = array([ [r, g, 150] for r, g in zip(50+100*abs(x), 50+100*abs(y)) ], dtype="uint8")
        TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,help"
        p = figure(title="Positions (x,y)", tools=TOOLS)
        p.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=1, line_color=None, name="PSO")
        p.xaxis.axis_label = 'X'
        p.yaxis.axis_label = 'Y'
        show(p)

if __name__ == "__PSO__":
    main()
function_index = int(3) #1-Exponencial 2-Rastrigin 3-Shaffer 4-Ackley
while (function_index < 1) or (function_index > 4):
    function_index = int(input("O índice da função precisa ser um número inteiro entre 1 e 4"))
pso_version_index = int(2) #1-SPSO 2-PSO-WL 3-PSO-WR [1] 4-Pso_chongpeng
while (pso_version_index < 1) or (pso_version_index > 4):
    pso_version_index = int(input("O índice da versão PSO precisa ser um número inteiro entre 1 e 3"))
options = {"pso_version":pso_version_index,
       "bounds": array([[-2, 2],[-2, 2]]), # input limits [(x_min,x_max)]
       "number_particles":int(30), # amount of particles
       "interactions":int(500), # amount of interactions that each particle will make
       "c1":float(1), # cognitive constant
       "c2":float(2), # social constant
       "wi":float(0.9), # initial inertia
       "wf":float(0.4), # final inertia
       "number_dimentions":int(2) # amount of dimentions
       }
pso = PSO(lambda x: myfunc(x,function_index, options["number_dimentions"]), options)
print(pso.velocity_average)
print(pso.gbest_mo)
print(pso.gbest)
print(pso.fit_gbest)

Graph(pso.history.position[1:,:])

#Graphics
X=pso.history.position[:,0]
Y=pso.history.position[:,1]
Fit=pso.history.fitness[:,0]

fig = go.Figure(data=[go.Scatter3d(
    x=X,
    y=Y,
    z=Fit,
    mode='markers',
    marker=dict(
        size=3,
        color=Fit,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])
fig.update_layout(  scene = dict(zaxis = dict(title= 'Fit')),
                    title = 'Fit versus Positions(x,y)',
                    margin=dict(r=0, l=0, b=0, t=30)
                  )
fig.show()

# Positions(x,y)
mp.figure()
mp.title('X versus Y')
mp.xlabel("X")
mp.ylabel("Y")
mp.plot(pso.history.position[1:,0],pso.history.position[1:,1],'.')
mp.show()
# Positions(x) X Fitness
mp.figure()
mp.title('X versus Fit')
mp.xlabel("X")
mp.ylabel("Fit")
mp.plot(pso.history.position[1:,0],pso.history.fitness[1:],'.')
mp.show()
# Positions(y) X Fitness
mp.figure()
mp.title('Y versus Fit')
mp.xlabel("Y")
mp.ylabel("Fit")
mp.plot(pso.history.position[1:,1],pso.history.fitness[1:],'.')
mp.show()
# Interactions(x) X Fitness(x)
mp.figure()
mp.title('Maxinter versus Fit')
mp.xlabel("Maxinter")
mp.ylabel("Fit")
mp.plot(pso.history.fitness[:],'.')
mp.show()
# Interactions(x) X Velocity(x)
mp.figure()
mp.title('Maxinter versus Velocity')
mp.xlabel("Maxinter")
mp.ylabel("Velocity")
mp.plot(pso.history.velocity[:,0],'.')
mp.show()

# References
# 1 - SANTANA, D. D. Inferência Estatística Clássica com Enxame de Partículas, Salvador: UFBA, 2014, p. 28–31.

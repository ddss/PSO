"""
@title : Particle Swarm Optimization (PSO)
@author: Renilton Ribeiro Almeida
"""

from numpy import array, append, random, vstack, diag, dot
from scipy.stats import qmc
import plotly.io as pio

pio.renderers.default = 'browser'

# --- MAIN ---------------------------------------------------------------------+


class Particle:

    def __init__(self, number_dimentions, bounds, **kwargs):
        """
        Class used to define the particles

        Parameters
        ----------
        number_dimentions: int
            amount of dimensions
        bounds: array
            position restriction
        **kwargs:
            non-mandatory parameters

        Attributes
        ----------
        c1 : int
            cognitive constant
        c2 : int
            social constant
        wi : float
            initial inertia weight
        wf : float
            final inertia weight
        pbest : array
            the best individual position
        fit_best : int
            the lowest individual value
        fit : list
            individual value
        position : array
            particle position
        velocity : array
            particle velocity

        Methods
        -------
        evaluate(myfunction)
            Evaluate the objetive function and record its value
        spso(g_best)
            Update particle velocity adding velocity cognitive and velocity social to velocity
        pso_wl(g_best, maxiter, i)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
        pso_wr(self, g_best)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
        pso_chongpeng(self, g_best, maxiter, i)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
        update_position(self, swarm, bounds)
            Update particle position based on new velocity updates
        """
        self.swarm = None
        self.c1 = kwargs.get("c1") if kwargs.get("c1") is not None else 1
        self.c2 = kwargs.get("c2") if kwargs.get("c2") is not None else 2
        self.wi = kwargs.get("wi") if kwargs.get("wi") is not None else 0.9
        self.wf = kwargs.get("wf") if kwargs.get("wf") is not None else 0.4
        peso_vel = kwargs.get("init_velocity") if kwargs.get("init_velocity") is not None else diag(random.rand(number_dimentions))
        peso_pos = kwargs.get("init_position") if kwargs.get("init_position") is not None else diag(random.rand(number_dimentions))
        self.pbest = array([[], []])
        self.fit_best = 10000
        self.fit = array([])

        # corrigir
        # velcocity e position como kwargs - validar a dimensão: nx1
        self.velocity = dot(peso_vel, (bounds[:, 1]-bounds[:, 0])) + (bounds[:, 0])
        self.position = dot(peso_pos, (bounds[:, 1]-bounds[:, 0])) + (bounds[:, 0])

    def evaluate(self, myfunction):
        """
        Evaluate the objetive function and record its value

        Parameters
        ----------
        myfunction: result of the function at a given point
        """
        self.fit = myfunction(self.position)
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
        """
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()
        w = 0.5 + (r3 / 2)
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def pso_chongpeng(self, g_best, maxiter, i):
        """
        Chongpeng [2] proposed a variant of PSO with non-linearly decreasing inertia weight
        where k1, k2 are two natural numbers, wi is the initial inertia weight, wf is the final value of weighting coefficient,
        maxiter is the maximum number of iteration and 'i' is current iteration.
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
        """
        r1 = random.random()
        r2 = random.random()
        k1 = random.randint(1)
        k2 = 1
        w = self.wf + ((self.wf - self.wi)*((1-(i/maxiter)**k1)**k2))
        vel_cognitive = self.c1 * r1 * (self.pbest - self.position)
        vel_social = self.c2 * r2 * (g_best - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def update_position(self, swarm, bounds):
        """
        Update particle position

        Parameters
        ----------
        swarm: array
            Group of particles with their positions, velocities and fitness
        bounds: array
            position restriction
        """
        self.position = self.position + self.velocity
        if abs(swarm.position[0]) < bounds[0][1] < abs(swarm.position[1]):
            swarm.position[1] = 2 * (swarm.position[1] / abs(swarm.position[1]))
            swarm.velocity = swarm.velocity * -0.1
        elif abs(swarm.position[0]) > bounds[0][1] > abs(swarm.position[1]):
            swarm.position[0] = 2 * (swarm.position[0] / abs(swarm.position[0]))
            swarm.velocity = swarm.velocity * -0.1
        elif abs(swarm.position[0]) > bounds[0][1] and abs(swarm.position[1]) > bounds[0][1]:
            swarm.position[0] = 2 * (swarm.position[0] / abs(swarm.position[0]))
            swarm.position[1] = 2 * (swarm.position[1] / abs(swarm.position[1]))
            swarm.velocity = swarm.velocity * -0.1


class PSO:

    class history:
        def __init__(self, dim):
            """
            Class that saves all positions, velocities, fitness and average velocities of each particle

            Parameters
            ----------
            dim : int
                amount of dimensions

            Attributes
            ----------
            self.position : array
                an array to record the positional coordinates of each particle in each interaction
            self.fitness : array
                an array to record the fitness of each particle in each interaction
            self.velocity : array
                an array to record the velocity coordinates of each particle in each interaction
            self.vel_average : array
                an array to record the velocity average in each interaction

            Methods
            -------
            add(self, position, fitness, velocity, vel_average)
                Makes an array with all the data for each parameter
            """
            self.position = array(random.rand(1, dim))
            self.fitness = array(random.rand(1, 1))
            self.velocity = array(random.rand(1, dim))
            self.vel_average = array(random.rand(1, dim))

        def add(self, position, fitness, velocity, vel_average):
            self.position = vstack([self.position, position])
            self.fitness = vstack([self.fitness, fitness])
            self.velocity = vstack([self.velocity, velocity])
            self.vel_average = vstack([self.vel_average, vel_average])

    def __init__(self, myfunction, pso_version, bounds, num_part, maxiter, **kwargs):
        """
        Class that uses particles in an optimization loop

        Parameters
        ----------
        myfunction: function
            result of the function at a given point
        pso_version: str or int
            version of pso to be used for optimization
        number_dimentions: int
            amount of dimensions
        bounds: array
            position restriction
        num_part:
            number of particles
        maxiter:
            maximum number of interactions
        **kwargs:
            non-mandatory parameters

        Attributes
        ----------
        self.fit_gbest : float
            best group value
        self.gbest : list
            best group position
        swarm : array
            establishes the swarm
        self.velocity_sum : float
            the sum of all velocities
        self.velocity_average : float
            the average of all velocities
        """
        number_dimentions = len(bounds)
        self.history = self.history(number_dimentions)
        self.fit_gbest = 10000
        self.gbest = []
        self.swarm = array([])
        sobol_velocity = qmc.Sobol(d=number_dimentions).random_base2(m=5)
        sobol_position = qmc.Sobol(d=number_dimentions).random_base2(m=5)
        for i in range(0, num_part):
            if kwargs.get("initial_swarm") == 'Sobol':
                peso_vel = diag(sobol_velocity[i, :])
                peso_pos = diag(sobol_position[i, :])
            else:
                peso_vel = None
                peso_pos = None
            self.swarm = append(self.swarm, Particle(number_dimentions, bounds, c1=kwargs.get('c1'), c2=kwargs.get('c2'),
                                                     wi=kwargs.get('wi'), wf=kwargs.get('wf'), init_velocity=peso_vel, init_position=peso_pos))
        i = 0
        p = 0
        g = 1
        self.velocity_sum = 0
        while i < maxiter and p < 1e3:  # stopping criteria
            for j in range(0, num_part):
                self.swarm[j].evaluate(myfunction)
                self.velocity_sum = self.velocity_sum + self.swarm[j].velocity
                self.velocity_average = (self.velocity_sum/g)
                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity, self.velocity_average)

                # determines if the current particle is the best (globally)
                if abs(self.swarm[j].fit-self.fit_gbest) < 1e-6:
                    p += 1

                if self.swarm[j].fit < self.fit_gbest:
                    self.gbest = list(self.swarm[j].position)
                    self.fit_gbest = float(self.swarm[j].fit)

                if pso_version == 1 or pso_version == 'SPSO':
                    self.swarm[j].spso(self.gbest)
                elif pso_version == 2 or pso_version == 'PSO-WL':
                    self.swarm[j].pso_wl(self.gbest, maxiter, i)
                elif pso_version == 3 or pso_version == 'PSO-WR':
                    self.swarm[j].pso_wr(self.gbest)
                elif pso_version == 4 or pso_version == 'pso_chongpeng':
                    self.swarm[j].pso_chongpeng(self.gbest, maxiter, i)
                self.swarm[j].update_position(self.swarm[j], bounds)
            g += 1
            i += 1
        # self.gbest_mo = self.gbest + (self.velocity_average*(random.uniform(bounds[0][0],bounds[0][1])))
        # self.fit_mo = myfunction(self.gbest_mo)
        # ----------------------------------------------------------------------------------------

"""
@title : Particle Swarm Optimization (PSO)
@author: Renilton Ribeiro Almeida
"""

from numpy import array, append, random, diag, dot, shape, where, reshape, vstack, full, linspace, sqrt, ndarray
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
        **kwargs: c1, c2, wi, wf, init_velocity, init_position
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
        self.c1 = kwargs.get("c1") if kwargs.get("c1") is not None else 1
        self.c2 = kwargs.get("c2") if kwargs.get("c2") is not None else 2
        self.wi = kwargs.get("wi") if kwargs.get("wi") is not None else 0.9
        self.wf = kwargs.get("wf") if kwargs.get("wf") is not None else 0.4
        peso_vel = kwargs.get("peso_vel") if kwargs.get("peso_vel") is not None else diag(random.rand(number_dimentions))
        peso_pos = kwargs.get("peso_pos") if kwargs.get("peso_pos") is not None else diag(random.rand(number_dimentions))
        self.pbest = array([])
        self.fit_best = float()
        self.fit = array([])
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
        if self.fit < self.fit_best or bool(self.fit_best) is False:
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
        xmax = bounds[:, 1]
        xmin = bounds[:, 0]
        testemax = swarm.position > xmax
        testemin = swarm.position < xmin
        swarm.velocity[testemax] = swarm.velocity[testemax] * -0.01
        swarm.velocity[testemin] = swarm.velocity[testemin] * -0.01
        swarm.position[testemax] = xmax[testemax]
        swarm.position[testemin] = xmin[testemin]


class PSO:

    @property
    def available_methods(self):
        return ['SPSO', 'PSO-WL', 'PSO-WR', 'PSO-Chonpeng']

    class history:
        def __init__(self, dim):
            """
            Class that saves all positions, velocities, fitness and average velocities of each particle
            Parameters
            ----------
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
            function_cut(self, restriction_max)
                Select only values below a predefined threshold
            """
            arr = [],
            self.position = ([])
            self.velocity = ([])
            self.vel_average = ([])
            self.fit_average = ([])
            self.fitness = ([])
            self.dim = dim
            self.v_sdesviation = ([])
            self.f_sdesviation = ([])
            for i in range(self.dim):
                self.position += arr
                self.velocity += arr

        def add(self, position, fitness, velocity):
            for i in range(0, self.dim):
                self.position[i] = append(self.position[i], position[i])
                self.velocity[i] = append(self.velocity[i], velocity[i])
            self.fitness = append(self.fitness, fitness)
        def map(self, restriction_max, function_cut):
            # TODO: incluir variáveis para fmax e function_cut
            # TODO: salvar em variáveis separadas
            if function_cut is not None:
                teste = function_cut
            else:
                fmax = full(shape(self.fitness), restriction_max)
                teste = (self.fitness <= fmax)
            self.fitness = self.fitness[teste]
            for i in range(self.dim):
                self.position[i] = self.position[i][where(teste)]
        def stack(self):
            self.sposition = array([self.position[0]])
            self.svelocity = array([self.velocity[0]])
            for i in range(1, self.dim):
                self.sposition = vstack((self.sposition, self.position[i]))
                self.svelocity = vstack((self.svelocity, self.velocity[i]))


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
        bounds: array with shape (nd x 2)
            position restriction
        num_part:
            number of particles
        maxiter: interge
            maximum number of interactions
        **kwargs: c1, c2, wi, wf, initial_swarm
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
        class validation:
            def __init__(self, bounds, num_part, maxiter, **kwargs):
                if type(bounds) != ndarray:
                    raise TypeError("The variable 'bounds' must be an array")
                elif type(num_part) != int:
                    raise TypeError("The variable 'num_part' must be an integer.")
                elif type(maxiter) != int:
                    raise TypeError("The variable 'maxiter' must be an integer.")
                elif shape(bounds)[1] != 2:
                    raise TypeError("The variable 'bounds' must have 2 columns")
        # TODO: VERIFICAR properties do python

        validation(bounds, num_part, maxiter)
        number_dimentions = bounds.shape[0]
        self.fit_gbest = 1e10
        self.gbest = []
        self.swarm = array([])
        self.restriction_max = kwargs.get("restriction_max")
        self.function_cut = kwargs.get("function_cut")

        if kwargs.get("initial_swarm") == 'Sobol':
            from scipy.stats import qmc
            init_sobol_vel = qmc.Sobol(d=number_dimentions).random_base2(m=round(sqrt(num_part)))
            init_sobol_pos = qmc.Sobol(d=number_dimentions).random_base2(m=round(sqrt(num_part)))
        if kwargs.get("initial_swarm") == 'Pattern':
            init_pattern_pos = array([linspace(0, 1, num_part), linspace(0, 1, num_part), linspace(1, 0, num_part), linspace(1, 0, num_part)])
            init_pattern_vel = array([linspace(0, 0.5, num_part), linspace(0.5, 0, num_part), linspace(0, 0.5, num_part), linspace(0.5, 0, num_part)])
        for i in range(0, num_part):
            if kwargs.get("initial_swarm") == 'Sobol':
                peso_velocity = diag(init_sobol_vel[i, :])
                peso_position = diag(init_sobol_pos[i, :])
            elif kwargs.get("initial_swarm") == 'Pattern':
                peso_velocity = diag(init_pattern_vel[:number_dimentions, i])
                peso_position = diag(init_pattern_pos[:number_dimentions, i])
            else:
                peso_velocity = None
                peso_position = None
            self.swarm = append(self.swarm, Particle(number_dimentions, bounds, c1=kwargs.get('c1'), c2=kwargs.get('c2'),
                                                     wi=kwargs.get('wi'), wf=kwargs.get('wf'), peso_vel=peso_velocity, peso_pos=peso_position))

        self.history = self.history(number_dimentions)
        i = 0
        k = 0

        self.stop_criterion = kwargs.get("stop_criterion") if kwargs.get("stop_criterion") is not None else 1e3
        self.significant_evolution = kwargs.get("significant_evolution") if kwargs.get("significant_evolution") is not None else 1e-6
        while i < maxiter and k < self.stop_criterion:  # stopping criteria
            for j in range(0, num_part):
                self.swarm[j].evaluate(myfunction)

                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)

                # determines if the current particle is the best (globally)
                if abs(self.swarm[j].fit-self.fit_gbest) < self.significant_evolution:
                    k += 1

                if self.swarm[j].fit < self.fit_gbest:
                    self.gbest = list(self.swarm[j].position)
                    self.fit_gbest = float(self.swarm[j].fit)

                if pso_version == 'SPSO':
                    self.swarm[j].spso(self.gbest)
                elif pso_version == 'PSO-WL':
                    self.swarm[j].pso_wl(self.gbest, maxiter, i)
                elif pso_version == 'PSO-WR':
                    self.swarm[j].pso_wr(self.gbest)
                elif pso_version == 'pso_chongpeng':
                    self.swarm[j].pso_chongpeng(self.gbest, maxiter, i)
                else:
                    raise NameError("Available PSO versions are: 'SPSO', 'PSO-WL', 'PSO-WR' and 'pso_chongpeng'.\nPlease choose one of them")
                # TODO: 'SWARM' não deveria entrar aqui como argumento
                self.swarm[j].update_position(self.swarm[j], bounds)
            i += 1
        if self.restriction_max or self.function_cut is not None:
            self.history.map(self.restriction_max, self.function_cut)
        self.history.stack()

        # self.gbest_mo = self.gbest + (self.velocity_average*(random.uniform(bounds[0][0],bounds[0][1])))
        # self.fit_mo = myfunction(self.gbest_mo)
        # ----------------------------------------------------------------------------------------

# References
# 1 - SANTANA, D. D. Inferência Estatística Clássica com Enxame de Partículas, Salvador: UFBA, 2014, p. 28–31.
# 2 - Huang Chongpeng, Zhang Yuling, Jiang Dingguo and Xu Baoguo, "On Some Non-linear Decreasing Inertia Weight Strategies
# in Particle Swarm Optimization*," in Proceedings of the 26th Chinese Control Conference, Zhangjiajie, Hunan, China, 2007, pp. 570-753.

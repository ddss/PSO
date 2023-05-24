"""
@title : Particle Swarm Optimization (PSO)
@author: Renilton Ribeiro Almeida
"""

from numpy import array, append, random, diag, dot, shape, where, reshape, hstack, vstack, \
    full, linspace, sqrt, ndarray, copy, argmin, argmax
import plotly.io as pio

pio.renderers.default = 'browser'


# --- MAIN ---------------------------------------------------------------------+
class Particle:

    def __init__(self, myfunction, number_dimensions, bounds, **kwargs):
        """
        Class used to define the particles

        Parameters
        ----------
        myfunction : function
            result of the function at a given point
        number_dimensions: int
            amount of dimensions
        bounds: array with shape (nd x 2)
            position restriction
        **kwargs:
            c1 : int
                cognitive constant
            c2 : int
                social constant
            wi : float
                initial inertia weight
            wf : float
                final inertia weight
            init_velocity : array with shape (nd x nd)
                diagonal array with values from a distribution or random
            init_position : array with shape (nd x nd)
                diagonal array with values from a distribution or random

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
        peso_vel : array with shape (nd x nd)
            diagonal array with values from a distribution or random
        peso_pos : array with shape (nd x nd)
            diagonal array with values from a distribution or random
        pbest : array with shape (1 x nd)
            the best individual position
        fit_best : float
            the lowest individual value
        fit : list
            individual value
        position : array with shape (1 x nd)
            particle position
        velocity : array with shape (1 x nd)
            particle velocity
        __function : function
            result of the function at a given point

        Methods
        -------
        evaluate(__function)
            Evaluate the objetive function and record its value
        spso(g_best)
            Update particle velocity adding cognitive velocity and social velocity to it
            SPSO (Standard Particle Swarm Optimizer) is a version of pso taken from the dissertation [1].
        pso_wl(g_best, maxiter, i)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
            PSO_WL (Linear Time-Varying inercia Weight Particle Swarm Optimizer) is a version of pso taken from the dissertation [1].
        pso_wr(self, g_best)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
            PSO_WR (Random Time-Varying Inertia Weight Particle Swarm Optimizer) is a version of pso taken from the dissertation [1].
        pso_chongpeng(self, g_best, maxiter, i)
            Update particle velocity multiplying velocity by inertia and adding velocity cognitive and velocity social to it
            PSO_CHONGPENG (Non-linear Decreasing Inertia Weight Particle Swarm Optimizer) is a version of pso taken from the article [2].

        update_position(self, vel_restraint)
            Update particle position based on new velocity updates

        References
        1 - SANTANA, D. D. Inferência Estatística Clássica com Enxame de Partículas, Salvador: UFBA, 2014, p. 28–31.
        2 - Huang Chongpeng, Zhang Yuling, Jiang Dingguo and Xu Baoguo, "On Some Non-linear Decreasing Inertia Weight Strategies
        in Particle Swarm Optimization*," in Proceedings of the 26th Chinese Control Conference, Zhangjiajie, Hunan, China, 2007, pp. 570-753.
        """

        self.c1 = kwargs.get("c1") if kwargs.get("c1") is not None else 1
        self.c2 = kwargs.get("c2") if kwargs.get("c2") is not None else 2
        self.wi = kwargs.get("wi") if kwargs.get("wi") is not None else 0.9
        self.wf = kwargs.get("wf") if kwargs.get("wf") is not None else 0.4
        peso_vel = kwargs.get("peso_vel") if kwargs.get("peso_vel") is not None else diag(random.rand(number_dimensions))
        peso_pos = kwargs.get("peso_pos") if kwargs.get("peso_pos") is not None else diag(random.rand(number_dimensions))
        self.bounds = bounds
        self.__function = myfunction
        self.pbest = array([])
        self.fit = ([])
        self.fit_best = float()
        self.velocity = dot(peso_vel, (bounds[:, 1]-bounds[:, 0])) + (bounds[:, 0])
        self.position = dot(peso_pos, (bounds[:, 1]-bounds[:, 0])) + (bounds[:, 0])

    def evaluate_min(self, inter):
        """
        Evaluate the objetive function and record its value. It also defines the pbest and fit_best attributes
        Parameters
        ----------
        inter: interaction number
        """
        self.fit = self.__function(self.position)
        if self.fit < self.fit_best or inter == 0:
            self.pbest = self.position
            self.fit_best = self.fit
    def evaluate_max(self, inter):
        """
        Evaluate the objetive function and record its value. It also defines the pbest and fit_best attributes
        Parameters
        ----------
        inter: interaction number
        """
        self.fit = self.__function(self.position)
        if self.fit > self.fit_best or inter == 0:
            self.pbest = self.position
            self.fit_best = self.fit

    def spso(self, g_best):
        """
        The SPSO (Standard Particle Swarm Optimizer)  update particle velocity
        adding velocity cognitive and velocity social to velocity
        Parameters
        ----------
        g_best: float
            the position value of all particles of the swarm
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

    def update_position(self, vel_restraint):
        """
        Update particle position

        Parameters
        ----------
        vel_restraint: float
            value by which the velocity will be multiplied when the particle touches the edges
        """
        self.position = self.position + self.velocity
        testemax = self.position > self.bounds[:, 1]
        testemin = self.position < self.bounds[:, 0]
        self.velocity[testemax] = self.velocity[testemax] * vel_restraint
        self.velocity[testemin] = self.velocity[testemin] * vel_restraint
        self.position[testemax] = self.bounds[:, 1][testemax]
        self.position[testemin] = self.bounds[:, 0][testemin]


class PSO:

    @property
    def available_methods(self):
        return ['SPSO', 'PSO-WL', 'PSO-WR', 'PSO-Chonpeng']

    class history:

        @property
        def fitness_region(self):
            return self._fitness[self.__test_region]

        @property
        def position_region(self):
            return self._position[:, self.__test_region]

        def __init__(self, number_dimensions, swarm, num_part):
            """
            Class that saves all positions, velocities, fitness and average velocities of each particle
            Parameters
            ----------
            number_dimensions : int
                number of dimensions
            Attributes
            ----------
            self._position : array with shape (nd x np*inter)
                an array to record the positional coordinates of each particle in each interaction
            self._velocity : array  with shape (nd x np*inter)
                an array to record the velocity coordinates of each particle in each interaction
            self._fitness : array with shape (1 x np*inter)
                an array to record the fitness of each particle in each interaction
            Methods
            -------
            add(self, position, fitness, velocity)
                Makes an array with all the data for each parameter
            region(self, function_cut)
                selects the graph below the fit value that was given for the cut
            """
            self.num_part = num_part
            self.number_dimensions = number_dimensions
            self._position = reshape(swarm[0].position, (self.number_dimensions, 1))
            self._velocity = reshape(swarm[0].velocity, (self.number_dimensions, 1))
            self._fitness = array(swarm[0].fit)
            for i in range(1, num_part):
                self._position = hstack((self._position, reshape(swarm[i].position, (self.number_dimensions, 1))))
                self._velocity = hstack((self._velocity, reshape(swarm[i].velocity, (self.number_dimensions, 1))))
                self._fitness = append(self._fitness, swarm[i].fit)

        def add(self, position, fitness, velocity):
            self._position = hstack((self._position, reshape(position, (self.number_dimensions, 1))))
            self._velocity = hstack((self._velocity, reshape(velocity, (self.number_dimensions, 1))))
            self._fitness = append(self._fitness, fitness)

        def region(self, function_cut, sense):
            cutter = full(shape(self._fitness), function_cut)
            if sense == 'up':
                self.__test_region = (self._fitness >= cutter)
            elif sense == 'down':
                self.__test_region = (self._fitness <= cutter)
            self.inter = int((shape(self.fitness_region)[0])/self.num_part)

    def __init__(self, myfunction, pso_version, bounds, num_part, maxiter, **kwargs):
        """
        Class that uses particles in an optimization loop
        Parameters
        ----------
        myfunction : function
            result of the function at a given point
        pso_version : str or int
            version of pso to be used for optimization
        bounds : array with shape (nd x 2)
            position restriction
        num_part : int
            number of particles
        maxiter : int
            maximum number of interactions
        **kwargs :
            c1 : int
                cognitive constant
            c2 : int
                social constant
            wi : float
                initial inertia weight
            wf : float
                final inertia weight, initial_swarm
            initial_swarm : str
                initialization type of velocities and positions
            function_cut : float
                fitness value for cutting the graph
            vel_restraint : float
                value by which the velocity will be multiplied when the particle crosses the edges
            sig_evolution_value : float
                minor significant evolution
            significant_evolution : int
                how many times an evolution less than minor significant evolution can be tolerated

        Attributes
        ----------
        number_dimensions: int
            amount of dimensions
        self.fit_gbest : float
            best group value
        self.gbest : list
            best group position
        self.swarm : array of particles with shape (1 x np)
            establishes the swarm
        self.function_cut : float
            fitness value for cutting the graph
        vel_restraint : float
            value by which the velocity will be multiplied when the particle crosses the edges.
            A small value is recommended so that the particle is not bouncing from one end to the
            other, and it must be negative so that the particle follows the path opposite to the
            end it is attached to.
        self.maxiter: int
            maximum number of interactions
        sig_evolution_value : float
            minor significant evolution
        significant_evolution : int
            how many times an evolution less than minor significant evolution can be tolerated
        """
        class validation:
            def __init__(self, bounds, num_part, maxiter, **kwargs):
                """
                Class that validates the parameters passed by the user
                Parameters
                ----------
                bounds: array with shape (nd x 2)
                    position restriction
                num_part: int
                    number of particles
                maxiter: int
                    maximum number of interactions
                **kwargs:
                """
                if type(bounds) != ndarray:
                    raise TypeError("The variable 'bounds' must be an array")
                if type(num_part) != int:
                    raise TypeError("The variable 'num_part' must be an integer.")
                if type(maxiter) != int:
                    raise TypeError("The variable 'maxiter' must be an integer.")
                if shape(bounds)[1] != 2:
                    raise TypeError("The variable 'bounds' must have 2 columns")
                if type(kwargs.get("c1")) != float and kwargs.get("c1") is not None:
                    raise TypeError("The variable 'c1' must be a number.")
                if type(kwargs.get("c2")) != float and kwargs.get("c2") is not None:
                    raise TypeError("The variable 'c2' must be a number.")
                if type(kwargs.get("wi")) != float and kwargs.get("wi") is not None:
                    raise TypeError("The variable 'wi' must be a number")
                if type(kwargs.get("wf")) != float and kwargs.get("wf") is not None:
                    raise TypeError("The variable 'wf' must be a number.")
                if type(kwargs.get("initial_swarm")) != str and kwargs.get("initial_swarm") is not None:
                    raise TypeError("The variable 'initial_swarm' must be a string.")
                if type(kwargs.get("function_cut")) != float and kwargs.get("function_cut") is not None:
                    raise TypeError("The variable 'initial_swarm' must be a number.")
                if type(kwargs.get("vel_restraint")) != float and kwargs.get("vel_restraint") is not None:
                    raise TypeError("The variable 'vel_restraint' must be a number.")
                if type(kwargs.get("sig_evolution_value")) != float and kwargs.get("sig_evolution_value") is not None:
                    raise TypeError("The variable 'sig_evolution_value' must be a number.")
                if type(kwargs.get("significant_evolution")) != int and kwargs.get("significant_evolution") is not None:
                    raise TypeError("The variable 'significant_evolution' must be an integer.")

        validation(bounds, num_part, maxiter, c1=kwargs.get("c1"), c2=kwargs.get("c2"), wi=kwargs.get("wi"), wf=kwargs.get("wf"),
                   initial_swarm=kwargs.get("initial_swarm"), vel_restraint=kwargs.get("vel_restraint"),
                   sig_evolution_value=kwargs.get("sig_evolution_value"), significant_evolution=kwargs.get("significant_evolution"))

        number_dimensions = bounds.shape[0]
        self.fit_gbest = float()
        self.gbest = []
        self.swarm = array([])
        self.function_cut = kwargs.get("function_cut")
        self.vel_restraint = kwargs.get("vel_restraint") if kwargs.get("vel_restraint") is not None else -0.01
        self.inter = 0
        self.bounds = bounds
        self.num_part = num_part
        self.pso_version = pso_version
        self.maxiter = maxiter if maxiter is not None else 1e3
        self.significant_evolution = kwargs.get("significant_evolution") if kwargs.get("significant_evolution") is not None else 1e3
        self.sig_evolution_value = kwargs.get("sig_evolution_value") if kwargs.get("sig_evolution_value") is not None else 1e-6

        if kwargs.get("initial_swarm") == 'Sobol':
            from scipy.stats import qmc
            init_sobol_vel = qmc.Sobol(d=number_dimensions).random_base2(m=round(sqrt(num_part)))
            init_sobol_pos = qmc.Sobol(d=number_dimensions).random_base2(m=round(sqrt(num_part)))
        if kwargs.get("initial_swarm") == 'Pattern':
            init_pattern_vel = linspace(1, 0, num_part)
            for i in range(1, number_dimensions):
                v = linspace((1-(i*0.1)), 0, num_part)
                init_pattern_vel = vstack((init_pattern_vel, v))
            p = linspace(0, 1, num_part),
            init_pattern_pos = array(p*number_dimensions)

        for i in range(0, num_part):
            if kwargs.get("initial_swarm") == 'Sobol':
                peso_velocity = diag(init_sobol_vel[i, :])
                peso_position = diag(init_sobol_pos[i, :])
            if kwargs.get("initial_swarm") == 'Pattern':
                peso_velocity = diag(init_pattern_vel[:number_dimensions, i])
                peso_position = diag(init_pattern_pos[:number_dimensions, i])
            else:
                peso_velocity = None
                peso_position = None
            self.swarm = append(self.swarm, Particle(myfunction, number_dimensions, self.bounds, c1=kwargs.get('c1'), c2=kwargs.get('c2'),
                                                     wi=kwargs.get('wi'), wf=kwargs.get('wf'), peso_vel=peso_velocity, peso_pos=peso_position))
            self.swarm[i].evaluate_min(0)

        self.history = self.history(number_dimensions, self.swarm, num_part)
    def map_min(self):
        k = 0
        while self.inter < self.maxiter and k < self.significant_evolution:  # stopping criteria
            gbest_previous = copy(self.fit_gbest)

            # determines if the current particle is the best (globally)
            if abs(gbest_previous - self.fit_gbest) < self.sig_evolution_value:
                k += 1
            else:
                k = 0

            for j in range(0, self.num_part):
                if self.swarm[j].fit < self.fit_gbest or self.inter == 0:
                    self.gbest = list(self.swarm[j].position)
                    self.fit_gbest = float(self.swarm[j].fit)

                if self.pso_version == 'SPSO':
                    self.swarm[j].spso(self.gbest)
                elif self.pso_version == 'PSO-WL':
                    self.swarm[j].pso_wl(self.gbest, self.maxiter, self.inter)
                elif self.pso_version == 'PSO-WR':
                    self.swarm[j].pso_wr(self.gbest)
                elif self.pso_version == 'pso_chongpeng':
                    self.swarm[j].pso_chongpeng(self.gbest, self.maxiter, self.inter)
                else:
                    raise NameError("Available PSO versions are: 'SPSO', 'PSO-WL', 'PSO-WR' and 'pso_chongpeng'.\nPlease choose one of them")
                self.swarm[j].update_position(self.vel_restraint)
                self.swarm[j].evaluate_min(self.inter)

                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)
            self.inter += 1

    def map_region(self):
        k = 0
        self.inter = 0
        self.fit_gbest = []
        while self.inter < self.maxiter and k < self.significant_evolution:  # stopping criteria
            gbest_previous = copy(self.fit_gbest)

                # determines if the current particle is the best (globally)
            if abs(gbest_previous - self.fit_gbest) < self.sig_evolution_value:
                k += 1
            else:
                k = 0

            for j in range(0, self.num_part):
                if self.inter == 0 or all(abs(self.bounds[:, 0] * 0.2) < abs(self.swarm[j].position)) and all(
                        abs(self.bounds[:, 1] * 0.2) < abs(self.swarm[j].position)):
                    if all(self.bounds[:, 0] != self.swarm[j].position) and all(
                            self.bounds[:, 1] != self.swarm[j].position):
                        self.gbest = list(self.swarm[j].position)
                        self.fit_gbest = float(self.swarm[j].fit)

                if self.pso_version == 'SPSO':
                    self.swarm[j].spso(self.gbest)
                elif self.pso_version == 'PSO-WL':
                    self.swarm[j].pso_wl(self.gbest, self.maxiter, self.inter)
                elif self.pso_version == 'PSO-WR':
                    self.swarm[j].pso_wr(self.gbest)
                elif self.pso_version == 'pso_chongpeng':
                    self.swarm[j].pso_chongpeng(self.gbest, self.maxiter, self.inter)
                else:
                    raise NameError(
                        "Available PSO versions are: 'SPSO', 'PSO-WL', 'PSO-WR' and 'pso_chongpeng'.\nPlease choose one of them")
                self.swarm[j].update_position(self.vel_restraint)
                self.swarm[j].evaluate_min(self.inter)

                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)
            self.inter += 1
            n = argmin(self.history._fitness)
            self.gbest = self.history._position[:, n]
            self.fit_gbest = self.history._fitness[n]
    def map_region2(self):
        k = 0
        self.inter = 0
        interval = linspace(0.2, 1.0, num=5)
        for i in range(0, 5):
            while self.inter < self.maxiter and k < self.significant_evolution:  # stopping criteria
                gbest_previous = copy(self.fit_gbest)
                # determines if the current particle is the best (globally)
                if abs(gbest_previous - self.fit_gbest) < self.sig_evolution_value:
                    k += 1
                else:
                    k = 0

                for j in range(0, self.num_part):
                    if self.inter == 0 or all(abs(self.bounds[:, 0] * interval[i]) > abs(self.swarm[j].position)) and all(
                            abs(self.bounds[:, 1] * interval[i]) > abs(self.swarm[j].position)):
                        if self.swarm[j].fit > self.fit_gbest or self.inter == 0:
                            self.gbest = list(self.swarm[j].position)
                            self.fit_gbest = float(self.swarm[j].fit)

                    if self.pso_version == 'SPSO':
                        self.swarm[j].spso(self.gbest)
                    elif self.pso_version == 'PSO-WL':
                        self.swarm[j].pso_wl(self.gbest, self.maxiter, self.inter)
                    elif self.pso_version == 'PSO-WR':
                        self.swarm[j].pso_wr(self.gbest)
                    elif self.pso_version == 'pso_chongpeng':
                        self.swarm[j].pso_chongpeng(self.gbest, self.maxiter, self.inter)
                    else:
                        raise NameError(
                            "Available PSO versions are: 'SPSO', 'PSO-WL', 'PSO-WR' and 'pso_chongpeng'.\nPlease choose one of them")
                    self.swarm[j].update_position(self.vel_restraint)
                    self.swarm[j].evaluate_max(self.inter)

                    self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)
                self.inter += 1
        n = argmin(self.history._fitness)
        self.gbest = self.history._position[:, n]
        self.fit_gbest = self.history._fitness[n]
    def map_max(self):
        k = 0
        self.inter = 0
        while self.inter < self.maxiter and k < self.significant_evolution:  # stopping criteria
            gbest_previous = copy(self.fit_gbest)

            # determines if the current particle is the best (globally)
            if abs(gbest_previous - self.fit_gbest) < self.sig_evolution_value:
                k += 1
            else:
                k = 0

            for j in range(0, self.num_part):
                if self.swarm[j].fit > self.fit_gbest or self.inter == 0:
                    self.gbest = list(self.swarm[j].position)
                    self.fit_gbest = float(self.swarm[j].fit)

                if self.pso_version == 'SPSO':
                    self.swarm[j].spso(self.gbest)
                elif self.pso_version == 'PSO-WL':
                    self.swarm[j].pso_wl(self.gbest, self.maxiter, self.inter)
                elif self.pso_version == 'PSO-WR':
                    self.swarm[j].pso_wr(self.gbest)
                elif self.pso_version == 'pso_chongpeng':
                    self.swarm[j].pso_chongpeng(self.gbest, self.maxiter, self.inter)
                else:
                    raise NameError(
                        "Available PSO versions are: 'SPSO', 'PSO-WL', 'PSO-WR' and 'pso_chongpeng'.\nPlease choose one of them")
                self.swarm[j].update_position(self.vel_restraint)
                self.swarm[j].evaluate_max(self.inter)

                self.history.add(self.swarm[j].position, self.swarm[j].fit, self.swarm[j].velocity)
            self.inter += 1

        n = argmax(self.history._fitness)
        self.gbest = self.history._position[:, n]
        self.fit_gbest = self.history._fitness[n]
        # ----------------------------------------------------------------------------------------

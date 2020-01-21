#!/usr/bin/env python

"""
Implementation of basic Differential Evolution Optimisation.
"""

import datetime
import math
import random


class Agent(object):
    """
    Encapsulation of a vector of parameters and their corresponding fitness.
    """

    def __init__(self, fitness_function):
        """
        Initialise a new instance of the class.
        :param fitness_function: The fitness function.
        :return: A new instance of the class.
        """
        assert hasattr(fitness_function, '__call__')

        self.__parameters = []
        self.__fitness = None
        self.__fitness_function = fitness_function

    def __getitem__(self, item):
        """
        Return the parameter at the specified index.
        :param item: The index of the parameter.
        :return: The parameter at the specified index.
        """
        return self.__parameters[item]

    def __len__(self):
        """
        Return the number of parameters.
        :return: The number of parameters.
        """
        return len(self.__parameters)

    def __setitem__(self, key, value):
        """
        Set the parameter with the specified index to the specified value.
        :param key: The index of the parameter.
        :param value: The new value for the parameter.
        """
        self.__parameters[key] = value
        self.__fitness = None

    def __str__(self):
        """
        Return a string representation of this instance.
        :return: A string representation of this instance.
        """
        return 'fitness:{0} parameters:{1}'.format(self.fitness, self.__parameters)

    def clone(self):
        """
        Return a copy of this instance.
        :return: A copy of this instance.
        """
        agent = Agent(self.__fitness_function)
        agent.__parameters = list(self.__parameters)
        agent.__fitness = self.__fitness
        return agent

    @property
    def fitness(self):
        """
        Return the fitness for this instance, executing the fitness function if necessary.
        :return: The fitness for this instance.
        """
        if self.__fitness is None:
            self.__fitness = self.__fitness_function(self.__parameters)
        return self.__fitness

    def randomize(self, parameter_count):
        """
        Randomize the parameters of this instance to values between 0 and 1.
        :param parameter_count: The number of parameters.
        """
        self.__parameters = [random.random() for _ in range(parameter_count)]
        self.__fitness = None


class Population(object):
    """
    Encapsulation of a population of agents.
    """

    def __init__(self, population_size, parameter_count, fitness_function):
        """
        Initialize a new instance of the class to random values.
        :param population_size: The number of agents in the population.
        :param parameter_count: The number of parameter per agent.
        :fitness_function: The fitness function.
        :return: A new instance of the class.
        """
        self.__agents = [Agent(fitness_function) for _ in range(population_size)]
        for agent in self.__agents:
            agent.randomize(parameter_count)

    def __getitem__(self, item):
        """
        Return the agent with the specified index.
        :param item: The index of the agent.
        :return: The agent with the specified index.
        """
        return self.__agents[item]

    def __len__(self):
        """
        Return the number of agents.
        :return: The number of agents.
        """
        return len(self.__agents)

    def __setitem__(self, key, value):
        """
        Set the agent with the specified index to the specified instance.
        :param key: The index of the agent.
        :param value: The new agent.
        """
        assert isinstance(value, Agent)
        self.__agents[key] = value

    def __str__(self):
        """
        Return a string representation of this instance.
        :return: A string representation of this instance.
        """
        return 'best:{0}'.format(self.best)

    @property
    def best(self):
        """
        Return the agent with the highest fitness value.
        :return: The agent with the highest fitness value.
        """
        return max(self.__agents, key=lambda x: x.fitness)


class ExitCondition(object):
    """
    A class providing descriptions for the reason a Differential Evolution Optimisation terminates.
    """
    ABORT = 'ABORT'
    ITERATIONS = 'ITERATIONS'
    TIMEOUT = 'TIMEOUT'


class Context(object):
    """
    Context information for the Differential Evolution optimizer; this instance is passed to the optional log method
    and eventually returned as a result of the optimization.
    """

    def __init__(self, optimizer):
        """
        Initialise a new instance of the class for the specified optimizer.
        :param optimizer: The optimizer for this instance.
        :return: A new instance of the class.
        """
        assert isinstance(optimizer, Optimizer)

        self.aborted = False
        self.elapsed = datetime.timedelta(seconds=0)
        self.exit_condition = None
        self.generation = 0
        self.optimizer = optimizer
        self.population = None
        self.start = datetime.datetime.now()

    @property
    def best(self):
        """
        Return the agent with the highest fitness value.
        :return: The agent with the highest fitness value.
        """
        return self.population.best


class Optimizer(object):
    """
    A Differential Evolution Optimizer.
    """

    def __init__(self):
        """
        Initialize a new instance of the class, setting the maximum number of iterations to 500, no maximum execution
        time, the population size to 100, the crossover probability to 50% and the differential weight to 1.0.
        :return: A new instance of the class.
        """
        self.crossover_probability = 0.5
        self.differential_weight = 1.0
        self.log = None
        self.max_iterations = 500
        self.population_size = 100
        self.timeout = None

    def maximize(self, fitness_function, parameter_count):
        """
        Attempt to find a maximum set of values for a specified fitness function; the number of dimensions in the
        problem domain is represented by the specified parameter count.
        :param fitness_function: The fitness function to maximize.
        :param parameter_count: the number of dimensions in the problem domain.
        :return: An instance of type Context that encapsulates the final state of the algorithm.
        """

        def fitness_function_wrapper(parameters):
            fitness = fitness_function(parameters)
            if math.isnan(fitness):
                fitness = float('-inf')
            return fitness

        # Initialize the population to random values.
        context = Context(self)
        context.population = Population(self.population_size, parameter_count, fitness_function_wrapper)

        # Loop until an exit condition occurs.
        while True:

            # Log the iteration until it's lost forever.
            context.elapsed = datetime.datetime.now() - context.start
            context.generation += 1
            self.__log(context)

            # Check for exit conditions.
            if context.aborted:
                context.exit_condition = ExitCondition.ABORT
                return context
            if self.max_iterations is not None and context.generation >= self.max_iterations:
                context.exit_condition = ExitCondition.ITERATIONS
                return context
            if self.timeout is not None and context.elapsed > self.timeout:
                context.exit_condition = ExitCondition.TIMEOUT
                return context

            def exclusive_random(selected):
                while True:
                    n = random.randint(0, self.population_size - 1)
                    if n not in selected:
                        return n

            # Loop over all agents in the population, each time selecting three other agents randomly.
            for agent_index_0 in range(self.population_size):
                agent_index_1 = exclusive_random([agent_index_0])
                agent_index_2 = exclusive_random([agent_index_0, agent_index_1])
                agent_index_3 = exclusive_random([agent_index_0, agent_index_1, agent_index_2])

                # Create a copy of the agent to see if it can be improved.
                original = context.population[agent_index_0]
                candidate = original.clone()

                # Get references to the other agents.
                agent_1 = context.population[agent_index_1]
                agent_2 = context.population[agent_index_2]
                agent_3 = context.population[agent_index_3]

                # Pick a random parameter index.
                r = random.randint(0, parameter_count)

                # Compute the agents new parameters.
                for i in range(parameter_count):
                    if i == r or random.random() < self.crossover_probability:
                        candidate[i] = agent_1[i] + self.differential_weight * (agent_2[i] - agent_3[i])

                # Keep the candidate if it offers an improvement over the previous agent.
                if candidate.fitness > original.fitness:
                    context.population[agent_index_0] = candidate

    def __log(self, context):
        if self.log is not None:
            self.log(context)

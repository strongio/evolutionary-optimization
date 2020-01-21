"""
Implementation of a basic Genetic Algorithm.
"""

import abc
import datetime
import math
import random


class Individual(object):
    """
    An individual in a population processed by the Genetic Algorithm.
    """

    def __init__(self, genome, fitness):
        """
        Initialise an individual in a population of the Genetic Algorithm.
        :param genome: The sequence of values representing the individual.
        :param fitness: The fitness of the individual, or NaN to indicate a very bad fitness.
        :return: An initialised individual.
        """
        assert genome is not None
        assert hasattr(genome, '__iter__')
        assert len(genome) >= 1
        assert all(type(genome[0]) is type(x) for x in genome)
        assert type(genome[0]) in (bool, int, float)

        assert fitness is not None

        self.__genome = tuple(genome)
        self.__fitness = float('-inf') if math.isnan(fitness) else fitness

    def __str__(self):
        """
        Return a string representing of this instance.
        :return: A string representing this instance.
        """
        def format_booleans():
            def format_bit(x):
                return '1' if x else '0'
            token = ''.join(map(format_bit, self.genome))
            return '{0}:[{1}]'.format(self.fitness, token)

        def format_numbers():
            token = ' '.join(map(str, self.genome))
            return '{0}:[{1}]'.format(self.fitness, token)

        return format_booleans() if type(self.genome[0]) is bool else format_numbers()

    @property
    def fitness(self):
        """
        Return the fitness associated with this instance.
        :return: The fitness associated with this instance.
        """
        return self.__fitness

    @property
    def genome(self):
        """
        Return the tuple of values associated with this instance.
        :return: A tuple of values associated with this instance.
        """
        return self.__genome


#######################################################################################################################
# INITIALISATION
#######################################################################################################################


class Initialisation(object):
    """
    A basic representation of the Initialisation phase of a Genetic Algorithm.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initialise(self, population_size, genome_length, fitness_function):
        """
        Initialise and return a population for a Genetic Algorithm.
        :param population_size: The size of the population.
        :param genome_length: The length of the genome for each individual in the population.
        :param fitness_function: The fitness function.
        :return: A list of individuals in the initial population of the Genetic Algorithm.
        """
        pass


class FixedInitialisation(Initialisation):
    """
    An implementation for Fixed Initialisation for the Initialisation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, setting the fixed initial value used for the genomes to 0.5.
        :return: A new instance of the class.
        """
        super(FixedInitialisation, self).__init__()
        self.initial_value = 0.5

    def initialise(self, population_size, genome_length, fitness_function):
        """
        Initialise and return a population for a Genetic Algorithm based on a Fixed Initialization.  Initialise each
        data point of each genome of each individual to initial_value; call the fitness function only once for the
        entire initial population because all individuals are the same.
        :param population_size: The size of the population.
        :param genome_length: The length of the genome for each individual in the population.
        :param fitness_function: The fitness function, evaluated once for the entire population.
        :return: A list of individuals in the initial population of the Genetic Algorithm.
        """
        assert 0.0 <= self.initial_value <= 1.0
        assert population_size > 0
        assert genome_length > 0

        genome = [self.initial_value] * genome_length
        fitness = fitness_function(genome)

        population = []
        while len(population) < population_size:
            individual = Individual(genome, fitness)
            population.append(individual)

        return population


class GaussianInitialisation(Initialisation):
    """
    An implementation for Gaussian Initialisation for the Initialisation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, setting the mu and sigma values to 0.5 and 0.15, respectively.
        :return: A new instance of the class.
        """
        super(GaussianInitialisation, self).__init__()
        self.random = random
        self.mu = 0.5
        self.sigma = 0.15

    def initialise(self, population_size, genome_length, fitness_function):
        """
        Initialise and return a population for a Genetic Algorithm based on a Gaussian Initialization.  Initialise each
        data point of each genome of each individual to a random Gaussian value; call the fitness function once for
        every member of the initial population.
        :param population_size: The size of the population.
        :param genome_length: The length of the genome for each individual in the population.
        :param fitness_function: The fitness function, evaluated once for every member of the initial population.
        :return: A list of individuals in the initial population of the Genetic Algorithm.
        """
        assert population_size > 0
        assert genome_length > 0

        def generate():
            rnd = self.random.gauss(self.mu, self.sigma)
            return min(max(0.0, rnd), 1.0)

        population = []
        while len(population) < population_size:
            genome = [generate() for _ in range(genome_length)]
            fitness = fitness_function(genome)
            individual = Individual(genome, fitness)
            population.append(individual)

        return population


class UniformInitialisation(Initialisation):
    """
    An implementation for Uniform Initialisation for the Initialisation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        super(UniformInitialisation, self).__init__()
        self.random = random

    def initialise(self, population_size, genome_length, fitness_function):
        """
        Initialise and return a population for a Genetic Algorithm based on a Uniform Initialization.  Initialise each
        data point of each genome of each individual to a random number between 0.0 and 1.0; call the fitness function
        once for every member of the initial population.
        :param population_size: The size of the population.
        :param genome_length: The length of the genome for each individual in the population.
        :param fitness_function: The fitness function, evaluated once for every member of the initial population.
        :return: A list of individuals in the initial population of the Genetic Algorithm.
        """
        assert population_size > 0
        assert genome_length > 0

        def generate():
            return self.random.uniform(0.0, 1.0)

        population = []
        while len(population) < population_size:
            genome = [generate() for _ in range(genome_length)]
            fitness = fitness_function(genome)
            individual = Individual(genome, fitness)
            population.append(individual)

        return population


#######################################################################################################################
# SELECTION
#######################################################################################################################

class Selection(object):
    """
    A basic representation of the Selection phase of a Genetic Algorithm.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Initialise and return a new instance of the class with the selection ratio set to 75%.
        :return: A new instance of the class.
        """
        self.selection_ratio = 0.75

    @abc.abstractmethod
    def select(self, population):
        """
        Select a subset of the specified population for breeding the next generation; the subset is the size of the
        population multiplied by the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: A subset of the population.
        """
        pass

    def compute_selection_size(self, population):
        """
        Compute and return the size of the subset of a specified population based on the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: The number of individuals to select for the subset.
        """
        assert 0 < self.selection_ratio <= 1.0
        size = max(1, int(round(float(len(population)) * self.selection_ratio)))
        assert 0 < size <= len(population)
        return size


class RouletteSelection(Selection):
    """
    A representation of the Roulette Selection phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        super(RouletteSelection, self).__init__()
        self.random = random

    def select(self, population):
        """
        Select a subset of the specified population for breeding the next generation; the subset is the size of the
        population multiplied by the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: A subset of the population.
        """
        size = self.compute_selection_size(population)
        assert all([p.fitness >= 0.0 for p in population])

        sigma = sum([p.fitness for p in population])

        breeders = []
        while len(breeders) < size:
            rnd = self.random.uniform(0.0, sigma)
            for individual in population:
                rnd -= individual.fitness
                if rnd <= 0.0:
                    breeders.append(individual)
                    break

        return breeders


class StochasticSelection(Selection):
    """
    A representation of the Stochastic Selection phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        super(StochasticSelection, self).__init__()
        self.random = random

    def select(self, population):
        """
        Select a subset of the specified population for breeding the next generation; the subset is the size of the
        population multiplied by the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: A subset of the population.
        """
        size = self.compute_selection_size(population)
        assert all([p.fitness >= 0.0 for p in population])

        sigma = sum([p.fitness for p in population])

        interval = sigma / float(size)
        rnd = self.random.uniform(0.0, interval)

        breeders = []
        while True:
            for individual in population:
                rnd -= individual.fitness
                while rnd <= 0.0:
                    breeders.append(individual)
                    if len(breeders) == size:
                        return breeders
                    rnd += interval


class TournamentSelection(Selection):
    """
    A representation of the Tournament Selection phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, setting the tournament size to 10%.
        :return: A new instance of the class.
        """
        super(TournamentSelection, self).__init__()
        self.random = random
        self.tournament_ratio = 0.1

    def select(self, population):
        """
        Select a subset of the specified population for breeding the next generation; the subset is the size of the
        population multiplied by the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: A subset of the population.
        """
        assert 0.0 <= self.tournament_ratio < 1.0

        size = self.compute_selection_size(population)
        tournament_size = int(round(len(population) * self.tournament_ratio))

        breeders = []
        while len(breeders) < size:

            a = self.random.randint(0, len(population) - tournament_size)
            b = a + tournament_size
            winner = population[a]

            for individual in population[a + 1:b]:
                if individual.fitness > winner.fitness:
                    winner = individual

            breeders.append(winner)

        return breeders


class TruncationSelection(Selection):
    """
    A representation of the Truncation Selection phase of a Genetic Algorithm.
    """

    def select(self, population):
        """
        Select a subset of the specified population for breeding the next generation; the subset is the size of the
        population multiplied by the selection ratio.
        :param population: The population from which to select a subset for breeding the next generation.
        :return: A subset of the population.
        """
        size = self.compute_selection_size(population)

        available = list(population)
        list.sort(available, key=lambda x: x.fitness, reverse=True)

        return available[:size]


#######################################################################################################################
# CROSSOVER
#######################################################################################################################


class Crossover(object):
    """
    A basic representation of the Crossover phase of a Genetic Algorithm.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def crossover(self, individuals):
        """
        Return a new 'genome' based on the two specified individuals; the genome returned can be evaluated with the
        fitness function before using it to initialise an individual in a population.
        :param individuals: A pair of individuals.
        :return: A new 'genome' based on the two specified individuals.
        """
        pass


class OnePointCrossover(Crossover):
    """
    A representation of the One Point Crossover phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        super(OnePointCrossover, self).__init__()
        self.random = random

    def crossover(self, individuals):
        """
        Return a new 'genome' based on the two specified individuals; the genome returned can be evaluated with the
        fitness function before using it to initialise an individual in a population.
        :param individuals: A pair of individuals.
        :return: A new 'genome' based on the two specified individuals.
        """
        assert len(individuals) == 2

        left, right = individuals
        assert isinstance(left, Individual)
        assert isinstance(right, Individual)

        n = len(left.genome)
        assert n == len(right.genome)
        assert n >= 2

        i = self.random.randint(1, n - 1)

        return list(left.genome[:i] + right.genome[i:])


class TwoPointCrossover(Crossover):
    """
    A representation of the Two Point Crossover phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        super(TwoPointCrossover, self).__init__()
        self.random = random

    def crossover(self, individuals):
        """
        Return a new 'genome' based on the two specified individuals; the genome returned can be evaluated with the
        fitness function before using it to initialise an individual in a population.
        :param individuals: A pair of individuals.
        :return: A new 'genome' based on the two specified individuals.
        """
        assert len(individuals) == 2

        outer, inner = individuals
        assert isinstance(outer, Individual)
        assert isinstance(inner, Individual)

        n = len(outer.genome)
        assert n == len(inner.genome)
        assert n >= 3

        i = self.random.randint(1, n - 2)
        j = self.random.randint(i + 1, n - 1)

        return list(outer.genome[:i] + inner.genome[i:j] + outer.genome[j:])


class UniformCrossover(Crossover):
    """
    A representation of the Uniform Crossover phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, setting the ratio used for the first parent to 50%.
        :return: A new instance of the class.
        """
        super(UniformCrossover, self).__init__()
        self.first_parent_ratio = 0.5
        self.random = random

    def crossover(self, individuals):
        """
        Return a new 'genome' based on the two specified individuals; the genome returned can be evaluated with the
        fitness function before using it to initialise an individual in a population.
        :param individuals: A pair of individuals.
        :return: A new 'genome' based on the two specified individuals.
        """
        assert 0.0 <= self.first_parent_ratio <= 1.0
        assert len(individuals) == 2

        first, second = individuals
        assert isinstance(first, Individual)
        assert isinstance(second, Individual)

        n = len(first.genome)
        assert n == len(second.genome)
        assert n >= 2

        counts = [0, 0]
        genome = []

        def choose(index):
            if self.random.uniform(0.0, 1.0) < self.first_parent_ratio:
                genome.append(first.genome[index])
                counts[0] += 1
            else:
                genome.append(second.genome[index])
                counts[1] += 1

        for i in range(n - 1):
            choose(i)

        if counts[0] == 0:
            genome.append(first.genome[n - 1])
        elif counts[1] == 0:
            genome.append(second.genome[n - 1])
        else:
            choose(n - 1)

        return genome


#######################################################################################################################
# MUTATION
#######################################################################################################################


class Mutation(object):
    """
    A basic representation of the Mutation phase of a Genetic Algorithm.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Initialise a new instance of the class, setting the point mutation ratio to 15%.
        :return:
        """
        self.point_mutation_ratio = 0.15

    @abc.abstractmethod
    def mutate(self, genome0):
        """
        Return a mutation of the specified genome.
        :param genome0: The genome before mutation; this method does not affect this list.
        :return: A new genome, possibly mutated.
        """
        pass


class BoundaryMutation(Mutation):
    """
    A  representation of the Boundary Mutation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return:
        """
        super(BoundaryMutation, self).__init__()
        self.random = random

    def mutate(self, genome0):
        """
        Return a mutation of the specified genome.
        :param genome0: The genome before mutation; this method does not affect this list.
        :return: A new genome, possibly mutated.
        """
        assert 0.0 <= self.point_mutation_ratio <= 1.0
        assert len(genome0) >= 1

        genome = []

        for allele in genome0:
            if self.random.uniform(0.0, 1.0) >= self.point_mutation_ratio:
                genome.append(allele)
                continue

            is_mutated = self.random.uniform(0.0, 1.0) < 0.5
            mutated_allele = 0.0 if is_mutated else 1.0
            genome.append(mutated_allele)

        return genome


class GaussianMutation(Mutation):
    """
    A  representation of the Gaussian Mutation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, setting the mu and sigma fields to 0.0 and 0.01, respectively.
        :return:
        """
        super(GaussianMutation, self).__init__()
        self.random = random
        self.mu = 0.0
        self.sigma = 0.01

    def mutate(self, genome0):
        """
        Return a mutation of the specified genome.
        :param genome0: The genome before mutation; this method does not affect this list.
        :return: A new genome, possibly mutated.
        """
        assert 0.0 <= self.point_mutation_ratio <= 1.0
        assert len(genome0) >= 1

        genome = []

        for allele in genome0:
            if self.random.uniform(0.0, 1.0) >= self.point_mutation_ratio:
                genome.append(allele)
                continue

            unbounded_mutated_allele = allele + self.random.gauss(self.mu, self.sigma)
            mutated_allele = min(max(0.0, unbounded_mutated_allele), 1.0)
            genome.append(mutated_allele)

        return genome


class UniformMutation(Mutation):
    """
    A  representation of the Uniform Mutation phase of a Genetic Algorithm.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return:
        """
        super(UniformMutation, self).__init__()
        self.random = random

    def mutate(self, genome0):
        """
        Return a mutation of the specified genome.
        :param genome0: The genome before mutation; this method does not affect this list.
        :return: A new genome, possibly mutated.
        """
        assert 0.0 <= self.point_mutation_ratio <= 1.0
        assert len(genome0) >= 1

        genome = []

        for allele in genome0:
            if self.random.uniform(0.0, 1.0) >= self.point_mutation_ratio:
                genome.append(allele)
                continue

            mutated_allele = self.random.uniform(0.0, 1.0)
            genome.append(mutated_allele)

        return genome


#######################################################################################################################
# OPTIMIZER
#######################################################################################################################

class ExitCondition(object):
    """
    A class providing descriptions for the reason a Genetic Algorithm terminates.
    """

    ABORT = 'ABORT'
    GENERATIONS = 'GENERATIONS'
    TIMEOUT = 'TIMEOUT'


class Context(object):
    """
    Context information for the Genetic Algorithm optimizer; this instance is passed to the optional log method and
    eventually returned as a result of the optimization.
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
        self.hall_of_fame = []
        self.optimizer = optimizer
        self.population = []
        self.start = datetime.datetime.now()

    def submit_to_hall_of_fame(self, individual, max_size):
        """
        Submits an individual to the hall of fame.
        :param individual: The individual submitted to the hall of fame.
        :param max_size: The maximum-allowed size of the hall of fame.
        :return: None
        """
        assert isinstance(individual, Individual)
        assert max_size > 0

        if individual in self.hall_of_fame:
            return

        self.hall_of_fame.append(individual)
        list.sort(self.hall_of_fame, key=lambda x: x.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:max_size]


class Optimizer(object):
    """
    A Genetic Algorithm optimizer.
    """

    def __init__(self):
        """
        Initialise a new instance of the class, with an initial population size of 100, an initial hall-of-fame size
        of 5, an initial elite count of 1, a maximum of 500 generations, and no maximum execution time.  By default,
        Uniform Initialization, Tournament Selection, One-Point Crossover, and Gaussian Mutation are used.
        :return: A new instance of the class.
        """
        self.mutation = GaussianMutation()
        self.crossover = OnePointCrossover()
        self.initialisation = UniformInitialisation()
        self.selection = TournamentSelection()

        self.population_size = 100
        self.hall_of_fame_size = 5
        self.elite_count = 1
        self.log = None
        self.max_iterations = 500
        self.timeout = None

    def maximize(self, fitness_function, genome_length):
        """
        Attempt to find a maximum set of values for a specified fitness function; the number of dimensions in the
        problem domain is represented by the specified length of 'genome'.
        :param fitness_function: The fitness function to maximize.
        :param genome_length: The length of the 'genome'; i.e. the number of dimensions in the problem domain.
        :return: An instance of type Context that encapsulates the final state of the algorithm.
        """
        assert hasattr(fitness_function, '__call__')
        assert genome_length > 0
        assert self.elite_count < self.population_size

        context = Context(self)

        # Create the initial population.
        context.population = self.initialisation.initialise(
            self.population_size,
            genome_length,
            fitness_function)

        # Update the hall of fame.
        for individual in context.population:
            context.submit_to_hall_of_fame(individual, self.hall_of_fame_size)

        # Loop until an exit condition occurs.
        while True:
            context.elapsed = datetime.datetime.now() - context.start
            context.generation += 1

            # Log the generation before it's lost forever.
            self.__log(context)
            if context.aborted:
                context.exit_condition = ExitCondition.ABORT
                return context
            if self.max_iterations is not None and context.generation >= self.max_iterations:
                context.exit_condition = ExitCondition.GENERATIONS
                return context
            if self.timeout is not None and context.elapsed > self.timeout:
                context.exit_condition = ExitCondition.TIMEOUT
                return context

            # Select breeders for the next generation.
            breeders = self.selection.select(context.population)
            assert len(breeders) >= 2
            list.sort(breeders, key=lambda x: x.fitness, reverse=True)

            # Select the elite, if any.
            elite = []
            for individual in sorted(context.population, key=lambda x: x.fitness, reverse=True):
                if len(elite) >= self.elite_count:
                    break
                elite.append(individual)

            # Prepare a new population that initially contains the elite.
            context.population = elite

            # Initialise indices for two non-adjacent breeders.
            i = 0
            j = len(breeders) // 2

            # Loop until enough offspring are produced.
            while len(context.population) < self.population_size:

                # Always perform crossover (recombination).
                genome = self.crossover.crossover(
                    sorted((breeders[i], breeders[j]), key=lambda x: x.fitness, reverse=True))

                # Perform mutation optionally.
                if self.mutation is not None:
                    genome = self.mutation.mutate(genome)

                # Add this individual to the population of the next generation.
                fitness = fitness_function(genome)
                individual = Individual(genome, fitness)
                context.population.append(individual)

                # Advance to the next breeders.
                i = (i + 1) % len(breeders)
                j = (j + 1) % len(breeders)

            # Update the hall of fame.
            for individual in context.population:
                context.submit_to_hall_of_fame(individual, self.hall_of_fame_size)

    def __log(self, context):
        if self.log is not None:
            self.log(context)

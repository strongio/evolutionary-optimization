from abc import ABCMeta, abstractmethod

import datetime
import math
import numpy
import random
import scipy.optimize

import GeneticAlgorithm
import ParticleSwarm
import DifferentialEvolutionAlgorithm


def _format_float(n, length=20, digits=8):
    """
    Return a float with a certain length for readability
    :param n: A float value
    :return: The float value padded with spaces
    """
    fmt = '{{:.{0}f}}'.format(digits)
    return '{{0: >{0}}}'.format(length).format(fmt.format(n))


def _format_text(n, length=20):
    """
    Return a string with a certain length for readability
    :param n: A string
    :return: The string padded with spaces or trimmed
    """
    return '{{0: >{0}}}'.format(length).format(n)[:length]


def _format_int(n, length=20):
    """
    Return an integer with a certain length for readability
    :param n: A integer value
    :return: The integer value padded with spaces
    """
    return '{{0: >{0}}}'.format(length).format(n)


def _lerp(percent, bounds):
    """
    Return a linearly interpreted value given a percentage and a range
    :param percent: The percentage
    :param bounds: The range
    :return: A linearly interpreted value given a percentage and a range
    """
    assert len(bounds) == 2
    lower, upper = bounds
    return ((upper - lower) * percent) + lower


def _get_param_in_exp(param, bounds):
    """
    Convert a log scaled parameter back to linear scale
    :param param: The log scaled parameter
    :param bounds: The parameter bounds in linear scale
    :return: The log scaled parameter scaled back
    """
    # assert len(bounds) == 2
    # assert bounds[0] >= 0.0
    # assert bounds[1] >= 0.0
    # log_bounds = [math.log(bounds[0]), math.log(bounds[1])]
    # param_in_log = _lerp(param, log_bounds)
    # return math.exp(param_in_log)
    return param


def _transform_genome(genome, param_bounds):
    """
    Transform the parameters from the optimization space back to normal
    :param genome: The list of parameters in optimization space
    :return: The list of parameters in their normal scales
    """
    assert len(genome) == len(param_bounds)
    return [_get_param_in_exp(genome[i], param_bounds[i]) for i in xrange(len(param_bounds))]


def _log(text):
    """
    Print text to standard output
    :param text: Text to be logged
    """
    print text


class Optimizer(object):
    """
    Base class for all optimizers
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def maximize(self, objective_function):
        """
        Return the optimized parameters given the objective function
        :param objective_function:
        :return: The optimal parameters
        """
        pass


class _GAOptimizer(Optimizer):
    """
    Encapsulate the Genetic Algorithm (ga) optimizer
    """

    def __init__(self, param_bounds):
        """
        Initialise the optimization environment
        :param param_bounds: Legal parameter bounds
        """
        super(_GAOptimizer, self).__init__()

        self.param_bounds = param_bounds
        self.num_params = len(self.param_bounds)
        self.optimizer = GeneticAlgorithm.Optimizer()
        self.optimizer.timeout = None
        self.optimizer.hall_of_fame_size = 5
        self.optimizer.max_iterations = 5
        self.optimizer.log = self.__log
        self.optimizer.elite_count = 1
        self.optimizer.population_size = 5

        self.optimizer.initialization = GeneticAlgorithm.UniformInitialisation()
        self.optimizer.selection = GeneticAlgorithm.TournamentSelection()
        self.optimizer.selection.tournament_ratio = 0.1
        self.optimizer.selection.selection_ratio = 0.75
        self.optimizer.crossover = GeneticAlgorithm.OnePointCrossover()
        self.optimizer.mutation = GeneticAlgorithm.GaussianMutation()
        self.optimizer.mutation.point_mutation_ratio = 0.15
        self.optimizer.mutation.mu = 0.0
        self.optimizer.mutation.sigma = 0.01

    def maximize(self, objective_function):
        """
        Attempt to find a maximum set of values for a specified fitness function
        :param objective_function: The objective function to maximize
        :return: The optimal parameters
        """

        _log('# algorithm            = {0}'.format(self.__class__.__name__))
        _log('# timeout              = {0}'.format(self.optimizer.timeout))
        _log('# elite_count          = {0}'.format(self.optimizer.elite_count))
        _log('# max_iterations       = {0}'.format(self.optimizer.max_iterations))
        _log('# population_size      = {0}'.format(self.optimizer.population_size))

        _log('# initialization       = {0}'.format(self.optimizer.initialization.__class__.__name__))
        _log('# selection            = {0}'.format(self.optimizer.selection.__class__.__name__))
        _log('# tournament_ratio     = {0}'.format(self.optimizer.selection.tournament_ratio))
        _log('# selection_ratio      = {0}'.format(self.optimizer.selection.selection_ratio))
        _log('# mutation             = {0}'.format(self.optimizer.mutation.__class__.__name__))
        _log('# point_mutation_ratio = {0}'.format(self.optimizer.mutation.point_mutation_ratio))
        _log('# mu                   = {0}'.format(self.optimizer.mutation.mu))
        _log('# sigma                = {0}'.format(self.optimizer.mutation.sigma))

        def fitness_function(genome):
            return objective_function(numpy.array(_transform_genome(genome, self.param_bounds)))

        context = self.optimizer.maximize(fitness_function, self.num_params)

        _log('#')
        _log('# exit_condition       = {0}'.format(context.exit_condition))
        _log('# generations          = {0}'.format(context.generation))

        return numpy.array(_transform_genome(
            context.hall_of_fame[0].genome, self.param_bounds))

    def __log(self, context):
        """
        Log for each generation
        :param context: optimizer context containing all information
        """
        assert isinstance(context, GeneticAlgorithm.Context)

        scores = [individual.fitness for individual in context.population]

        _log('#')
        _log('# POPULATION FOR GENERATION {0}'.format(context.generation))
        _log('# average_fitness = {0}'.format(sum(scores) / len(context.population)))
        _log('# min_fitness     = {0}'.format(min(scores)))
        _log('# max_fitness     = {0}'.format(max(scores)))
        _log('#')
        _log('# {0} {1} {2} {3}'.format(
            _format_text('gen', 3),
            _format_text('idv', 3),
            _format_text('fitness'),
            ' '.join([_format_text('param{0}'.format(i)) for i in xrange(self.num_params)])))

        for index, individual in enumerate(sorted(context.population, key=lambda x: -x.fitness)):
            assert isinstance(individual, GeneticAlgorithm.Individual)
            parameters = _transform_genome(individual.genome, self.param_bounds)

            _log('{0} {1} {2} {3}'.format(
                _format_int(context.generation, 5),
                _format_int(index + 1, 3),
                _format_float(individual.fitness),
                ' '.join([_format_float(param) for param in parameters])))


class _SOptimizer(Optimizer):
    """
    Encapsulate Scipy optimizers
    """

    def __init__(self, param_bounds, option):
        """
        Initialize the optimization environment
        :param param_bounds: Legal parameter bounds
        """
        assert option in ['nm', 'bfgs', 'pow', 'bh', 'sdea']
        super(_SOptimizer, self).__init__()

        self.option = option
        self.timeout = None
        self.max_executions = 1
        self.param_bounds = param_bounds
        self.num_params = len(self.param_bounds)

    def maximize(self, objective_function):
        """
        Attempt to find a maximum set of values for a specified fitness function
        :param objective_function: The objective function to maximize
        :return: The optimal parameters
        """
        assert (self.timeout is not None) or (self.max_executions is not None)

        _log('# algorithm            = {0}'.format(self.__class__.__name__))
        _log('# timeout              = {0}'.format(self.timeout))
        _log('# max_executions       = {0}'.format(self.max_executions))

        def make_random_parameters():
            def make_random_parameter((low, high)):
                return random.uniform(low, high)

            return numpy.array([
                make_random_parameter(self.param_bounds[ii])
                for ii in xrange(self.num_params)
            ])

        end_time = None if self.timeout is None \
            else datetime.datetime.now() + self.timeout

        names = ['param{0}'.format(i) for i in xrange(self.num_params)]
        _log('# {0}'.format('\t'.join(
            map(str, ['execution', 'state', 'fitness'] + names)
        )))

        execution = 0

        def log_with_score(*args):
            columns = [execution, args[0], objective_function(args[1])]
            columns.extend(args[1])
            _log('\t'.join(map(str, columns)))

        if self.option != 'sdea':
            best_parameters = make_random_parameters()
            best_score = objective_function(best_parameters)
            log_with_score('init', best_parameters)
        else:
            best_score = -float('inf')

        while True:
            execution += 1

            if self.max_executions is not None:
                if execution > self.max_executions:
                    return best_parameters
            if end_time is not None:
                if datetime.datetime.now() > end_time:
                    return best_parameters

            random_parameters = make_random_parameters()

            log_with_score('{0}-in'.format(self.option), random_parameters)

            if self.option in ['nm', 'bfgs', 'pow']:
                method = {
                    'nm': 'nelder-mead', 'bfgs': 'BFGS', 'pow': 'POWELL'
                }[self.option]
                parameters = scipy.optimize.minimize(
                    lambda x: -objective_function(x),
                    random_parameters,
                    method=method,
                    callback=lambda x: log_with_score(
                        '{0}-iter'.format(self.option), x
                    ),
                )
            elif self.option == 'bh':
                parameters = scipy.optimize.basinhopping(
                    lambda x: -objective_function(x),
                    random_parameters,
                    callback=lambda x, y, z: log_with_score('bh-iter', x, y, z),
                )
            else:
                parameters = scipy.optimize.differential_evolution(
                    lambda x: -objective_function(x),
                    bounds=self.param_bounds,
                    callback=lambda x, convergence: log_with_score(
                        'sdea-iter', x, convergence
                    ),
                )
            parameters = parameters.x

            log_with_score('{0}-out'.format(self.option), parameters)

            score = objective_function(parameters)

            if score > best_score:
                best_score = score
                best_parameters = parameters


class _PSOptimizer(Optimizer):
    """
    Encapsulate the Particle Swarm Optimizer (pso)
    """

    def __init__(self, param_bounds):
        """
        Initialize the optimization environment
        :param param_bounds: Legal parameter bounds
        """
        super(_PSOptimizer, self).__init__()

        self.param_bounds = param_bounds
        self.num_params = len(self.param_bounds)
        self.optimizer = ParticleSwarm.Optimizer()
        self.optimizer.log = self.__log
        self.optimizer.omega = 0.9
        self.optimizer.max_initial_velocity = 0.02
        self.optimizer.population_size = 50
        self.optimizer.max_iterations = 50
        self.optimizer.phi_particle = 0.3
        self.optimizer.phi_swarm = 0.1

    def maximize(self, objective_function):
        """
        Attempt to find a maximum set of values for a specified fitness function
        :param objective_function: The objective function to maximize
        :return: The optimal parameters
        """
        _log('# algorithm            = {0}'.format(self.__class__.__name__))
        _log('# timeout              = {0}'.format(self.optimizer.timeout))
        _log('# max_iterations       = {0}'.format(self.optimizer.max_iterations))
        _log('# population_size      = {0}'.format(self.optimizer.population_size))
        _log('# max_initial_velocity = {0}'.format(self.optimizer.max_initial_velocity))
        _log('# omega                = {0}'.format(self.optimizer.omega))
        _log('# phi_particle         = {0}'.format(self.optimizer.phi_particle))
        _log('# phi_swarm            = {0}'.format(self.optimizer.phi_swarm))

        def fitness_function(parameters):
            return objective_function(numpy.array(_transform_genome(parameters, self.param_bounds)))

        context = self.optimizer.maximize(fitness_function, self.num_params)

        _log('#')
        _log('# exit_condition       = {0}'.format(context.exit_condition))
        _log('# iterations           = {0}'.format(context.iteration))
        _log('# execution_time       = {0}'.format(context.elapsed))

        return numpy.array(_transform_genome(context.best.positions, self.param_bounds))

    def __log(self, context):
        """
        Log for each generation
        :param context: optimizer context containing all information
        """
        assert isinstance(context, ParticleSwarm.Context)

        current_scores = [particle.current.fitness for particle in context.particles]
        best_scores = [particle.best.fitness for particle in context.particles]

        _log('#')
        _log('# PARTICLES FOR ITERATION {0}'.format(context.iteration))
        _log('# swarm_fitness           = {0}'.format(context.best.fitness))
        _log('# best_average_fitness    = {0}'.format(sum(best_scores) / len(best_scores)))
        _log('# best_minimum_fitness    = {0}'.format(min(best_scores)))
        _log('# best_maximum_fitness    = {0}'.format(max(best_scores)))
        _log('# current_average_fitness = {0}'.format(sum(current_scores) / len(current_scores)))
        _log('# current_minimum_fitness = {0}'.format(min(current_scores)))
        _log('# current_maximum_fitness = {0}'.format(max(current_scores)))
        _log('#')

        _log('# {0} {1} {2} {3} {4} {5}'.format(
            _format_text('gen', 3),
            _format_text('idv', 3),
            _format_text('fitness'),
            ' '.join([_format_text('param{0}'.format(i)) for i in xrange(self.num_params)]),
            _format_text('best_fitness'),
            ' '.join([_format_text('best-param{0}'.format(i)) for i in xrange(self.num_params)])))

        for index, particle in enumerate(sorted(context.particles, key=lambda p: -p.best.fitness)):
            current_positions = _transform_genome(particle.current.positions, self.param_bounds)
            best_positions = _transform_genome(particle.best.positions, self.param_bounds)
            _log('{0} {1} {2} {3} {4} {5}'.format(
                _format_int(context.iteration, 5),
                _format_int(index, 3),
                _format_float(particle.current.fitness),
                ' '.join([_format_float(x) for x in current_positions]),
                _format_float(particle.best.fitness),
                ' '.join([_format_float(x) for x in best_positions])))


class _DEAOptimizer(Optimizer):
    """
    Encapsulate the Differential Evolution Algorithm (dea) optimizer
    """

    def __init__(self, param_bounds):
        """
        Initialize the optimization environment
        :param param_bounds: Legal parameter bounds
        """
        super(_DEAOptimizer, self).__init__()
        self.param_bounds = param_bounds
        self.num_params = len(self.param_bounds)
        self.optimizer = DifferentialEvolutionAlgorithm.Optimizer()
        self.optimizer.timeout = None
        self.optimizer.crossover_probability = 0.5
        self.optimizer.differential_weight = 1.0
        self.optimizer.log = self.__log
        self.optimizer.population_size = 5
        self.optimizer.max_iterations = 5

    def maximize(self, objective_function):
        """
        Attempt to find a maximum set of values for a specified fitness function
        :param objective_function: The objective function to maximize
        :return: The optimal parameters
        """

        _log('# algorithm             = {0}'.format(self.__class__.__name__))
        _log('# timeout               = {0}'.format(self.optimizer.timeout))
        _log('# max_iterations        = {0}'.format(self.optimizer.max_iterations))
        _log('# population_size       = {0}'.format(self.optimizer.population_size))
        _log('# crossover_probability = {0}'.format(self.optimizer.crossover_probability))
        _log('# differential_weight   = {0}'.format(self.optimizer.differential_weight))

        def fitness_function(genome):
            return objective_function(numpy.array(_transform_genome(genome, self.param_bounds)))

        context = self.optimizer.maximize(fitness_function, self.num_params)

        _log('#')
        _log('# exit_condition       = {0}'.format(context.exit_condition))
        _log('# generations          = {0}'.format(context.generation))

        return numpy.array(_transform_genome(
            context.population.best, self.param_bounds))

    def __log(self, context):
        """
        Log for each generation
        :param context: optimizer context containing all information
        """
        assert isinstance(context, DifferentialEvolutionAlgorithm.Context)

        scores = [individual.fitness for individual in context.population]

        _log('#')
        _log('# POPULATION FOR GENERATION {0}'.format(context.generation))
        _log('# average_fitness = {0}'.format(sum(scores) / len(context.population)))
        _log('# min_fitness     = {0}'.format(min(scores)))
        _log('# max_fitness     = {0}'.format(max(scores)))
        _log('#')
        _log('# {0} {1} {2} {3}'.format(
            _format_text('gen', 3),
            _format_text('idv', 3),
            _format_text('fitness'),
            ' '.join([_format_text('param{0}'.format(i)) for i in xrange(self.num_params)])))

        for index, individual in enumerate(sorted(context.population, key=lambda a: -a.fitness)):
            parameters = _transform_genome(individual, self.param_bounds)

            _log('{0} {1} {2} {3}'.format(
                _format_int(context.generation, 5),
                _format_int(index + 1, 3),
                _format_float(individual.fitness),
                ' '.join([_format_float(param) for param in parameters])))


def create(option='nm', k=10, param_count=5):
    """
    Prepare the a specific optimizer
    :param options: The model and optimizer options
    """

    # def lower_upper(target):
    #     return math.exp(math.log(target) - math.log(k)), target * k
    # param_bounds = map(lower_upper, targets)

    param_bounds = [[-1e6, +1e6] for _ in range(param_count)]

    if option in ['nm', 'bfgs', 'pow', 'bh', 'sdea']:
        optimizer = _SOptimizer(param_bounds, option)

    elif option == 'ga':
        optimizer = _GAOptimizer(param_bounds)

    elif option == 'pso':
        optimizer = _PSOptimizer(param_bounds)

    elif option == 'dea':
        optimizer = _DEAOptimizer(param_bounds)

    else:
        assert False, 'Unsupported optimization method {0}'.format(option)

    _log('#')
    _log('# {0}'.format(datetime.datetime.now()))
    _log('#')
    return optimizer

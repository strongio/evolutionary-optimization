#!/usr/bin/env python

"""
Implementation of landscape functions and experiments used for Strong blog.
"""

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
import evopt.OptimizerFactory
import sys
import cStringIO

plt.rcParams["font.family"] = "Linux Libertine Display"


def rastrigin(X):
    A = 10
    return -(A * len(X) + \
           sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X]))


def sphere(X):
    return -sum([x**2 for x in X])


def rosenbrock(X):
    return -sum([100 * (X[i+1] - X[i]**2)**2 + (1 - X[i])**2 for i in range(len(X) - 1)])


def styblinski_tang(X):
    return -0.5 * sum([x**4 - (16 * x**2) + (5 * 5) for x in X])


def alpine_one(X):
    return -sum([abs(x * np.sin(x) + 0.1 * x) for x in X])


FUNCS = {
    'rastrigin': rastrigin,
    'sphere': sphere,
    'rosenbrock': rosenbrock,
    'styblinski_tang': styblinski_tang,
    'alpine_one': alpine_one
}

OPTIONS = ['pso', 'ga', 'dea', 'nm', 'bfgs', 'pow', 'bh', 'sdea']


def plot_functions(func='rastrigin'):
    def func_group(*X, **kwargs):
        func = kwargs.get('func', None)
        return func(X)
    assert func in FUNCS.keys()
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(X, Y)
    Z = func_group(X, Y, func=FUNCS[func])

    fig = plt.figure(figsize=(3, 3))
    ax = fig.gca(projection='3d')

    ax.plot_surface(
        X, Y, Z,
        rstride=1,
        cstride=1,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=False
    )
    for i in range(6):
        angle = 15 + 360 / 6 * i
        ax.view_init(30, angle)
        plt.tight_layout()
        plt.savefig('./images/3d-{0}-{1}.png'.format(func, angle))


def _test_plot():
    for key in FUNCS.keys():
        plot_functions(func=key)


def _test_run(option, quiet=True, land_func=rastrigin, param_count=5):
    stdout_ = sys.stdout
    stream = cStringIO.StringIO()
    sys.stdout = stream

    optimizer = OptimizerFactory.create(option=option, param_count=param_count)
    if option in ['pso', 'ga', 'dea']:
        optimizer.optimizer.population_size = 100
        optimizer.optimizer.max_iterations = 500
    best_params = optimizer.maximize(land_func)
    best_fitness = land_func(best_params)

    sys.stdout = stdout_
    lines = stream.getvalue().split('\n')

    if not quiet:
        for line in lines:
            print(line)

    return best_fitness, best_params


def _test_batch(exp_size=3, param_count=5):
    to_plot = []
    for func_name in FUNCS.keys():
        land_func = FUNCS[func_name]
        results = {}
        for option in OPTIONS:
            results[option] = []
            for _ in range(exp_size):
                results[option].append(
                    _test_run(
                        option,
                        quiet=True,
                        land_func=land_func,
                        param_count=param_count
                    )[0]
                )
        to_plot.append((func_name, results))

    print(to_plot)

    def plot_boxes(to_plot, zoom=False, title='test.pdf'):
        plt.figure(figsize=(18, 4))
        for i, result_pair in enumerate(to_plot):
            func_name, results = result_pair
            plt.subplot(1, len(FUNCS), i + 1)
            labels = OPTIONS
            data = [results[option] for option in OPTIONS]
            labels = [e.upper() for e in labels]
            plt.boxplot(data)
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.xlabel(func_name)
            plt.ylabel('f(X), maximization objective')
            plt.axvline(x=3.5, label='EA (left) Scipy (right)')
            plt.legend()
            plt.grid()

            if zoom:
                means = [np.mean(to_plot[i][1][e]) for e in OPTIONS]
                stds = [np.std(to_plot[i][1][e]) for e in OPTIONS]
                best_mean = max(means)
                best_std = stds[means.index(max(means))]
                plt.ylim(best_mean - 20 * best_std, best_mean + 20 * best_std)

            plt.tight_layout()
        plt.savefig(title)

    def plot_boxes_pairs(to_plot):
        for i, result_pair in enumerate(to_plot):
            plt.figure(figsize=(8, 4))

            func_name, results = result_pair
            labels = OPTIONS
            data = [results[option] for option in OPTIONS]

            plt.subplot(1, 2, 1)
            plt.boxplot(data)

            labels = [e.upper() for e in labels]
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.xlabel('{0} zoomed out'.format(func_name))
            plt.ylabel('f(X), maximization objective')
            plt.axvline(x=3.5, label='EA (left) Scipy (right)')
            plt.legend()
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.boxplot(data)
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.xlabel('{0} zoomed in'.format(func_name))
            plt.ylabel('f(X), maximization objective')
            means = [np.mean(to_plot[i][1][e]) for e in OPTIONS]
            stds = [np.std(to_plot[i][1][e]) for e in OPTIONS]
            best_mean = max(means)
            best_std = stds[means.index(max(means))]
            plt.ylim(best_mean - 20 * best_std, best_mean + 20 * best_std)
            plt.axvline(x=3.5, label='EA (left) Scipy (right)')
            plt.legend()
            plt.grid()

            plt.tight_layout()

            plt.savefig('./images/pair_{0}.png'.format(func_name))

    plot_boxes(to_plot, zoom=False, title='./images/batch.png')
    plot_boxes(to_plot, zoom=True, title='./images/batch-zoom.png')
    plot_boxes_pairs(to_plot)


if __name__ == '__main__':
    # option = 'sdea'
    # optimizer = OptimizerFactory.create(option=option, param_count=4)
    # if option in ['pso', 'ga', 'dea']:
    #     optimizer.optimizer.max_iterations = 3
    #     optimizer.optimizer.population_size = 5
    # result = optimizer.maximize(rosenbrock)
    # print(result)

    _test_batch(exp_size=10, param_count=8)
    # _test_plot()


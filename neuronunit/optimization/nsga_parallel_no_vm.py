##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##

import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('Agg')

import sys
import os
import ipyparallel as ipp

rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
dview = rc[:]
def set_pickle_protocol(protocol):
    import pickle
    pickle.DEFAULT_PROTOCOL = protocol
    import cloudpickle
    cloudpickle.DEFAULT_PROTOCOL = protocol
    from ipyparallel.serialize import serialize
    serialize.PICKLE_PROTOCOL = protocol
#    raise Exception(serialize.PICKLE_PROTOCOL)

dview.apply_sync(set_pickle_protocol,4)
rc[:].use_cloudpickle()
inv_pid_map = {}


with dview.sync_imports(): # Causes each of these things to be imported on the workers as well as here.
    import matplotlib
    matplotlib.use('Agg')
    import get_neab
    import neuronunit
    import model_parameters as modelp
    import pdb
    import array
    import random
    import sys

    import numpy as np
    import matplotlib.pyplot as plt
    import quantities as pq
    from deap import algorithms
    from deap import base
    from deap.benchmarks.tools import diversity, convergence
    from deap import creator
    from deap import tools


    import quantities as qt
    import os, sys
    import os.path

    import deap as deap


    import quantities as pq
    import neuronunit.capabilities as cap
    history = tools.History()
    import numpy as np

    import sciunit
    thisnu = str(os.getcwd())+'/../..'
    sys.path.insert(0,thisnu)
    import sciunit.scores as scores




def p_imports():
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
    model.load_model()
    return

dview.apply_sync(p_imports)
p_imports()
from deap import base
from deap import creator
toolbox = base.Toolbox()

class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.results=None
        self.name=''
        self.attrs = {}
        self.params=None
        self.score=None
        self.fitness=None
        self.lookup={}
        self.rheobase=None
        self.fitness = creator.FitnessMax

with dview.sync_imports():

    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
    NDIM = len(BOUND_UP)+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)


def p_imports():
    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
    NDIM = len(BOUND_UP)+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    return
dview.apply_sync(p_imports)

BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
NDIM = len(BOUND_UP)+1 #One extra to store rheobase values in.

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
toolbox.register("select", tools.selNSGA2)




def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params




def update_pop(pop, trans_dict):

    from itertools import repeat
    import numpy as np
    import copy
    pop = [toolbox.clone(i) for i in pop ]
    #import utilities
    def transform(ind):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        model.load_model()
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        model.set_run_params(**param_dict)
        return model

    if len(pop) > 0:
        models = dview.map_sync(transform, pop)
    else:
        models = transform(pop)
    return models
##
# Start of the Genetic Algorithm
# For good results, MU the size of the gene pool
# should at least be as big as number of dimensions/model parameters
# explored.
##

MU = 10
NGEN = 7
CXPB = 0.9

import numpy as np
pf = tools.ParetoFront()

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
stats.register("avg", np.mean)
stats.register("std", np.std)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "min", "max", "avg", "std"

dview.push({'pf':pf})
trans_dict = get_trans_dict(param_dict)
td = trans_dict
dview.push({'trans_dict':trans_dict,'td':td})

pop = toolbox.population(n = MU)

pop = [ toolbox.clone(i) for i in pop ]

models = update_pop(pop, td)

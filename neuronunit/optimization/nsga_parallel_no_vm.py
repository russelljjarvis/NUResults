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
#from ipyparallel import depend, require, dependent

#profile_hook = cProfile.Profile()
#atexit.register(ProfExit, profile_hook)
#profile_hook.enable()
rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
rc[:].use_cloudpickle()
inv_pid_map = {}
dview = rc[:]
lview = rc.load_balanced_view()
ar = rc[:].apply_async(os.getpid)
pids = ar.get_dict()
inv_pid_map = pids
pid_map = {}

#Map PIDs onto unique numeric global identifiers via a dedicated dictionary
for k,v in inv_pid_map.items():
    pid_map[v] = k

with dview.sync_imports(): # Causes each of these things to be imported on the workers as well as here.
    import get_neab
    import matplotlib
    import neuronunit
    import model_parameters as modelp
    try:
        matplotlib.use('Qt5Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
    except:
        matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
                          # on the worker threads.
    import pdb
    import array
    import random
    import sys

    import numpy as np
    import matplotlib.pyplot as plt
    import quantities as pq
    from deap import algorithms
    from deap import base
    from deap.benchmarks.tools import diversity, convergence, hypervolume
    from deap import creator
    from deap import tools


    import quantities as qt
    import os, sys
    import os.path

    import deap as deap
    import functools
    import utilities
    vm = utilities.VirtualModel()



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
    print(get_neab.LEMS_MODEL_PATH)
    new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
    print(new_file_path)

    os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
    model = ReducedModel(new_file_path,name='vanilla',backend='NEURON')
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

def vm_to_ind(vm,td):
    '''
    Re instanting Virtual Model at every update vmpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''

    ind =[]
    for k in td.keys():
        ind.append(vm.attrs[td[k]])
    ind.append(vm.rheobase)


    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel

    new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
    model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
    model.load_model()
    model.update_run_params(vms.attrs)
    return ind



def update_pop(pop, trans_dict):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    from itertools import repeat
    import numpy as np
    import copy
    pop = [toolbox.clone(i) for i in pop ]
    #import utilities
    def transform(ind):
        '''
        Re instanting Virtual Model at every update vmpop
        is Noneifying its score attribute, and possibly causing a
        performance bottle neck.
        '''

        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        model.load_model()
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        model.update_run_params(param_dict)
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

vmpop = update_pop(pop, td)

# sometimes done in serial in order to get access to opaque stdout/stderr
#fitnesses = []
#for v in vmpop:
#   fitnesses.append(evaluate(v))

import copy
fitnesses = dview.map_sync(evaluate, copy.copy(vmpop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


pop = tools.selNSGA2(pop, MU)
# only update the history after crowding distance has been assigned
history.update(pop)


### After an evaluation of error its appropriate to display error statistics
#pf = tools.ParetoFront()
pf.update([toolbox.clone(i) for i in pop])
record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

def difference(unit_predictions):
    unit_observations = get_neab.tests[0].observation['value']
    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    return float(unit_delta)

verbose = False
means = np.array(logbook.select('avg'))
gen = 1
rh_mean_status = np.mean([ v.rheobase for v in vmpop ])
rhdiff = difference(rh_mean_status * pq.pA)
verbose = True
while (gen < NGEN and means[-1] > 0.05):
    gen += 1
    offspring = tools.selNSGA2(pop, len(pop))
    if verbose:
        for ind in offspring:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights[0]))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues[0]))
            #print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid[0]))
            print(rhdiff)
    offspring = [toolbox.clone(ind) for ind in offspring]

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        # deleting the fitness values is what renders them invalid.
        # The invalidness is used as a flag for recalculating them.
        # Their fitneess needs deleting since the attributes which generated these values have been mutated
        # and hence they need recalculating
        # Mutation also implies breeding, if a gene is mutated it was also recently recombined.
        del ind1.fitness.values, ind2.fitness.values

    invalid_ind = []
    for ind in offspring:
        if ind.fitness.valid == False:
            invalid_ind.append(ind)
    # Need to make sure that update_vm_pop does not replace instances of the same model
    # Thus waisting computation.
    vmoffspring = update_vm_pop(copy.copy(invalid_ind), trans_dict) #(copy.copy(invalid_ind), td)
    vmoffspring , _ = check_rheobase(copy.copy(vmoffspring))
    rh_mean_status = np.mean([ v.rheobase for v in vmoffspring ])
    rhdiff = difference(rh_mean_status * pq.pA)
    print('the difference: {0}'.format(difference(rh_mean_status * pq.pA)))
    # sometimes fitness is assigned in serial, although slow gives access to otherwise hidden
    # stderr/stdout
    # fitnesses = []
    # for v in vmoffspring:
    #    fitness.append(evaluate(v))
    fitnesses = list(dview.map_sync(toolbox.evaluate, copy.copy(vmoffspring)))
    mf = np.mean(fitnesses)

    for ind, fit in zip(copy.copy(invalid_ind), fitnesses):
        ind.fitness.values = fit
        if verbose:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues))
            print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid))

    # Its possible that the offspring are worse than the parents of the penultimate generation
    # Its very likely for an offspring population to be less fit than their parents when the pop size
    # is less than the number of parameters explored. However this effect should stabelize after a
    # few generations, after which the population will have explored and learned significant error gradients.
    # Selecting from a gene pool of offspring and parents accomodates for that possibility.
    # There are two selection stages as per the NSGA example.
    # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    # pop = toolbox.select(pop + offspring, MU)

    # keys = history.genealogy_tree.keys()
    # Grab evaluated history items and chuck them into the mixture.
    # We want to select among the best from the whole history of the GA, not just penultimate and present generations.
    # all_hist = [ history.genealogy_history[i] for i in keys if history.genealogy_history[i].fitness.valid == True ]
    # pop = tools.selNSGA2(offspring + all_hist, MU)

    pop = tools.selNSGA2(offspring + pop, MU)

    record = stats.compile(pop)
    history.update(pop)

    logbook.record(gen=gen, evals=len(pop), **record)
    pf.update([toolbox.clone(i) for i in pop])
    means = np.array(logbook.select('avg'))
    pf_mean = np.mean([ i.fitness.values for i in pf ])


    # if the means are not decreasing at least as an overall trend something is wrong.
    print('means from logbook: {0} from manual meaning the fitness: {1}'.format(means,mf))
    print('means: {0} pareto_front first: {1} pf_mean {2}'.format(logbook.select('avg'), \
                                                        np.sum(np.mean(pf[0].fitness.values)),\
                                                        pf_mean))


import pickle
with open('complete_dump.p','wb') as handle:
   pickle.dump([vmpop,pop,pf,history,logbook],handle)
'''
lists = pickle.load(open('complete_dump.p','rb'))
vmpop,pop,pf,history,logbook = lists[4],lists[3],lists[2],lists[1],lists[0]
'''

import net_graph
vmhistory = update_vm_pop(history.genealogy_history.values(),td)
net_graph.plotly_graph(history,vmhistory)
#net_graph.graph_s(history)
net_graph.plot_log(logbook)
net_graph.just_mean(logbook)
net_graph.plot_objectives_history(logbook)

#Although the pareto front surely contains the best candidate it cannot contain the worst, only history can.
best_ind_dict_vm = update_vm_pop(pf[0:2],td)
best_ind_dict_vm , _ = check_rheobase(best_ind_dict_vm)

best, worst = net_graph.best_worst(history)
listss = [best , worst]
best_worst = update_vm_pop(listss,td)
best_worst , _ = check_rheobase(best_worst)

print(best_worst[0].attrs,' == ', best_ind_dict_vm[0].attrs, ' ? should be the same (eyeball)')
print(best_worst[0].fitness.values,' == ', best_ind_dict_vm[0].fitness.values, ' ? should be the same (eyeball)')

# This operation converts the population of virtual models back to DEAP individuals
# Except that there is now an added 11th dimension for rheobase.
# This is not done in the general GA algorithm, since adding an extra dimensionality that the GA
# doesn't utilize causes a DEAP error, which is reasonable.

net_graph.plot_evaluate( best_worst[0],best_worst[1])
net_graph.plot_db(best_worst[0],name='best')
net_graph.plot_db(best_worst[1],name='worst')

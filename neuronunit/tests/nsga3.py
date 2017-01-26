'''
import os
os.system('ipcluster start --profile=jovyan --debug &')
os.system('sleep 5')
import ipyparallel as ipp
rc = ipp.Client(profile='jovyan')
print('hello from before cpu ')
print(rc.ids)
#quit()
v = rc.load_balanced_view()
'''
import time
init_start=time.time()
import get_neab

"""
Code from the deap framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
Conversion to its parallel form took two lines:
from scoop import futures
"""
import array
import random
import json

import numpy as np

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
# -1.0, -1.0, -1.0, -1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.sciunitscore=[]
        self.model=None
        self.error=None
        self.results=None
        self.name=''
        self.attrs={}
        self.params=None
        self.score=None
        self.fitness=None
        self.s_html=None
toolbox = base.Toolbox()

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

#param=['vr','a','b']
#rov0 = np.linspace(-65,-55,2)
#rov1 = np.linspace(0.015,0.045,2)
#rov2 = np.linspace(-0.0010,-0.0035,2)
#These are  dimension
#rov0 = np.linspace(-65,-55,1000)
#rov1 = np.linspace(0.015,0.045,1000)
#rov2 = np.linspace(-0.0010,-0.0035,1000)

'''
h.m_RS_RS_pop[i].v0 = -60.0
h.m_RS_RS_pop[i].k = 7.0E-4
h.m_RS_RS_pop[i].vr = -60.0
h.m_RS_RS_pop[i].vt = -40.0
h.m_RS_RS_pop[i].vpeak = 35.0
h.m_RS_RS_pop[i].a = 0.030000001
h.m_RS_RS_pop[i].b = -0.0019999999
h.m_RS_RS_pop[i].c = -50.0
h.m_RS_RS_pop[i].d = 0.1
h.m_RS_RS_pop[i].C = 1.00000005E-4
'''
NDIM= 10
rov=[]

vr = np.linspace(-75,-45,1000)
a = np.linspace(0.015,0.045,1000)
b = np.linspace(-0.0010,-0.0035,1000)

k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,1000)
c = np.linspace(-50.0-10.0,-50+10,1000)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,1000)
d = np.linspace(0.050,0.2,1000)
v0 = np.linspace(-75,-45,1000)
vt =  np.linspace(-50,-30,1000)
vpeak = np.linspace(25,40,1000)
param=['vr','a','b','C','c','d','v0','k','vt','vpeak']

rov.append(a)
rov.append(b)
rov.append(c)
rov.append(C)
rov.append(d)
rov.append(k)


rov.append(vr)
rov.append(v0)
rov.append(vt)
rov.append(vpeak)


'''

rov.append(rov0)
rov.append(rov1)
rov.append(rov2)
'''
seed_in=1

BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
NDIM = len(rov)
LOCAL_RESULTS=[]
import functools

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
import deap as deap

toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
model.local_run()

def evaluate(individual):#This method must be pickle-able for scoop to work.
    for i, p in enumerate(param):
        name_value=str(individual[i])
        #reformate values.
        model.name=name_value
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value

    individual.attrs=attrs
    b4nrncall=time.time()
    model.update_run_params(attrs)
    afternrncall=time.time()
    LOCAL_RESULTS.append(afternrncall-b4nrncall)
    individual.params=[]
    for i in attrs['//izhikevich2007Cell'].values():
        if hasattr(individual,'params'):
            individual.params.append(i)

    individual.results=model.results
    score = get_neab.suite.judge(model)
    import numpy as np
    individual.error = [ np.abs(i.score) for i in score.unstack() ]
    individual.s_html=score.to_html()
    error=individual.error
    assert individual.results
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],



toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

toolbox.register("map", futures.map)


def plotss(pop,gen):
    import matplotlib.pyplot as plt
    plt.clf()

    for ind in pop:
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'])
            plt.xlabel(str(ind.attrs))
            #str(scoop.worker)+
    plt.savefig('snap_shot_at_'+str(gen)+'.png')
    #plt.hold(False)
    plt.clf()
    #return 0

def main(seed=None):

    random.seed(seed)

    NGEN=10
    MU=20

    CXPB = 0.9
    import numpy as numpy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(evaluate, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        print(gen)
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        #assert ind.results

        offspring = [toolbox.clone(ind) for ind in offspring]
        #print('cloning not true clone')


        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        #import pdb
        #pdb.set_trace()

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        plotss(invalid_ind,gen)

            # Select the next generation population
        #This way the initial genes keep getting added to each generation.
        #pop = toolbox.select(pop + offspring, MU)
        #This way each generations genes are completely replaced by the result of mating.
        pop = toolbox.select(offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        pop.sort(key=lambda x: x.fitness.values)
        import numpy
        front = numpy.array([ind.fitness.values for ind in pop])
        plt.scatter(front[:,0], front[:,1], front[:,2], front[:,3])
        plt.axis("tight")
        plt.savefig('front.png')
        plt.clf()
    f=open('stats_summart.txt','w')
    f.write(list(logbook))

    f=open('mean_call_length.txt','w')
    f.write(np.mean(LOCAL_RESULTS))
    f.write('the number of calls to NEURON on one CPU only')
    f.write(len(LOCAL_RESULTS))

    pop=list(pop)
    plt.clf()
    plt.hold(True)
    for i in logbook:
        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)
    #'{}{}'.format("finish_time: ",finish_time)
    return pop, list(logbook)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    start_time=time.time()
    whole_initialisation=start_time-init_start

    pop, stats = main()
    f=open('html_score_matrix.html','w')
    f.write(pop[0].s_html)
    #s_html
    finish_time=time.time()
    ga_time=finish_time-start_time
    plt.clf()
    print(stats)
    f=open('stats_summart.txt','w')
    f.write(stats)
    #print(LOCAL_RESULTS)
    plt.clf()
    plt.hold(True)

    #pdb.set_trace()
    for i in stats:

        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)
    '{}{}'.format("finish_time: ",finish_time)

    plt.clf()
    #import pdb
    #pdb.set_trace()
    #plotss(invalid_ind,gen)
    '''
    plotr=LOCAL_RESULTS[len(LOCAL_RESULTS)-1]
    plt.plot(plotr['t'],plotr['vm'])
    plt.savefig('final_results_from_only_one_CPU.png')

    plt.clf()
    '''
    #NGEN=4
    plotss(pop,NGEN)

    #plt


    # plt.show()

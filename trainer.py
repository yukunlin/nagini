import random
from logger import *
from mlp import *
from pendulum import *
from numpy import *
from multiprocessing import Pool

random.seed()

ORGANISIMS = 50
TOP = 11
RATIO_MUTANTS = 0.1
NEURON_COUNT = [2, 10, 10, 1]
FUNCTS = ["tansig", "tansig", "tansig", "purelin"]
ITERATIONS = 500
WEIGHT_RANGE = 5.0

def breed(mlpA, mlpB):
    weightsA = mlpA.weights
    biasesA  = mlpA.bias
    weightsB = mlpB.weights
    biasesB  = mlpB.bias

    proportionA = random.random()
    proportionB = 1.0 - proportionA

    newWeights = []
    newBiases = []

    for i in range(len(weightsA)):
        newWeights.append(proportionA * weightsA[i] + proportionB * weightsB[i])
        newBiases.append(proportionA * biasesA[i] + proportionB * biasesB[i])

    child = MLP(2, NEURON_COUNT, FUNCTS)
    child.weights = newWeights
    child.bias = newBiases

    return child

def crossBreed(L):
    nextGeneration = []
    count = 0

    for i in range(TOP):
        parent1 = L[i]
        for j in range(i+1, len(L), 1):
            count += 1
            parent2 = L[j]
            mutate = False
            if random.random() < RATIO_MUTANTS:
                #print ("Mutate")
                mutate = True

            if mutate:
                nextGeneration.append(mutation())
            else:
                nextGeneration.append(breed(parent1, parent2))

            if count == ORGANISIMS:
                return nextGeneration

    return nextGeneration

def sortByFitness(Lpair):
    return sorted(Lpair, cmp = lambda x, y : 1 if x[0] > y[0] else -1)


def mutation():
    mutant = MLP(2,NEURON_COUNT, FUNCTS)
    mutant.genWB(WEIGHT_RANGE)
    return mutant

def testOrganism((mlp, pendulum, steps)):
    errs =[]

    for x in range(steps):
        mlpInputs = list(pendulum.rotational)
        mlpInputs = array(map(lambda x: [x], mlpInputs))
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]

        pendulum.update(mlpControl)
        error = abs(pendulum.rotational[0]-pi) + abs(pendulum.rotational[1])
        if (x == steps - 1):
            errs.append(error)

    return [sum(errs), mlp]

def outputControls(mlp, pendulum, steps):
    controls = []
    for x in range(steps):
        mlpInputs = list(pendulum.rotational)
        mlpInputs = array(map(lambda x: [x], mlpInputs))
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]
        pendulum.update(mlpControl)
        print(pendulum.rotational)
        
        controls.append(mlpControl)

    return controls

def testPopulation(population, pendulums):
    p = Pool(8)
    args = []

    for x in range(ORGANISIMS):
        args.append((population[x], pendulums[x], ITERATIONS))
    Lpairs = p.map(testOrganism, args)

    sortedPairs = sortByFitness(Lpairs)
    print map(lambda x : x[0], sortedPairs )[0]

    p.close()
    return map(lambda x : x[1], sortedPairs )

def populate():
    organisms = []
    for x in range(ORGANISIMS):
        org = MLP(2, NEURON_COUNT, FUNCTS)
        org.genWB(WEIGHT_RANGE)
        organisms.append(org)

    return organisms

def generatePendulums():
    pendulums = []
    for x in range(ORGANISIMS):
        pendulums.append(InvertedPendulum())

    return pendulums


if __name__ == "__main__":
    population = populate()
    pendulums = generatePendulums()
    epoch = 20
    best = None

    for x in range(epoch):
        sortedPopulation = testPopulation(population, pendulums)
        if x == epoch - 1:
            best = sortedPopulation[0]

        population = crossBreed(sortedPopulation)
        pendulums = generatePendulums()

    newPend = InvertedPendulum()
    bestControls = outputControls(best,newPend,ITERATIONS)

    #print bestControls

    log = Logger("controls.csv")
    log.write(bestControls)
    log.close()

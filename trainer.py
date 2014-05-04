import random
from mlp import *
from pendulum import *
from numpy import *

random.seed()

ORGANISIMS = 50
RATIO_MUTANTS = 0.1
NEURON_COUNT = [2, 10, 10, 1]
FUNCTS = ["logsig", "logsig", "logsig", "purelin"]

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

    child = MLP(NEURON_COUNT, FUNCTS)
    child.weights = newWeights
    child.bias = newBiases

    return child

def crossBreed(L):
    nextGeneration = []
    count = 0

    for i in range(len(L)):
        parent1 = L[i]
        for j in range(i+1, len(L), 1):
            count += 1
            parent2 = L[j]
            mutate = False

            if random.random() < RATIO_MUTANTS:
                mutate = True

            if mutate:
                nextGeneration.append(mutation())
            else:
                nextGeneration.append(breed(parent1, parent2))

            if count == ORGANISIMS:
                break

    return nextGeneration

def sortByFitness(Lpair):
    return reverse(sort(Lpair))

def mutation():
    mutant = MLP(2,NEURON_COUNT, FUNCTS)
    mutant.genWB(20.0)
    return mutant

def testNetwork(mlp, pendulum, steps):
    errs =[]

    for x in range(steps):
        mlpInputs = asmatrix(pendulum.rotational)
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]

        pendulum.update(mlpControl)
        error = abs(pendulum.rotational[0]) + abs(pendulum.rotational[1])
        err.append(error)

    return sum(errs)

def populate():
    organisms = []
    for x in range(ORGANISIMS):
        org = MLP(2, NEURON_COUNT, FUNCTS)
        organisms.append(org)

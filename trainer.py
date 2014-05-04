import random
from mlp import *
from pendulum import *
from numpy import *

random.seed()

ORGANISIMS = 50
RATIO_MUTANTS = 0.1
NEURON_COUNT = [2, 10, 10, 1]
FUNCTS = ["tansig", "tansig", "tansig", "purelin"]
ITERATIONS = 5

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
                return nextGeneration

    return nextGeneration

def sortByFitness(Lpair):
    return sorted(Lpair, cmp = lambda x, y : 1 if x[0] > y[0] else -1)


def mutation():
    mutant = MLP(2,NEURON_COUNT, FUNCTS)
    mutant.genWB(20.0)
    return mutant

def testOrganism(mlp, pendulum, steps):
    errs =[]

    for x in range(steps):
        mlpInputs = list(pendulum.rotational)
        mlpInputs = array(map(lambda x: [x], mlpInputs))
        print(mlpInputs)
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]

        pendulum.update(mlpControl)
        error = abs(pendulum.rotational[0]) + abs(pendulum.rotational[1])
        errs.append(error)

    return [sum(errs), mlp]

def testPopulation(population, pendulums):
    Lpairs = []
    for x in range(ORGANISIMS):
        Lpairs.append(testOrganism(population[x],pendulums[x],100))

    sortedPairs = sortByFitness(Lpairs)
    print map(lambda x : x[0], sortedPairs )

    return map(lambda x : x[1], sortedPairs )

def populate():
    organisms = []
    for x in range(ORGANISIMS):
        org = MLP(2, NEURON_COUNT, FUNCTS)
        org.genWB(20)
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
 #   print(testPopulation(population,pendulums))

    for x in range(ITERATIONS):
        sortedPopulation = testPopulation(population, pendulums)
        population = crossBreed(sortedPopulation)
        print(len(population))
        pendulums = generatePendulums()


import random
from logger import *
from mlp import *
from pendulum import *
from numpy import *
from multiprocessing import Pool

random.seed()

ORGANISIMS = 100
TOP = 15
RATIO_MUTANTS = 0.15
NEURON_COUNT = [2, 15, 15, 1]
FUNCTS = ["tansig", "tansig", "tansig", "purelin"]
ITERATIONS = 1000
WEIGHT_RANGE = 5.0
EPOCH = 2000
POOLS = 64

def breed(mlpA, mlpB):
    weightsA = mlpA.weights
    biasesA  = mlpA.bias
    weightsB = mlpB.weights
    biasesB  = mlpB.bias

    proportionA = random.random()
    proportionB = 1.0 - proportionA

    newWeights = []
    newBiases = []

    if random.random() < RATIO_MUTANTS:
        mutant = mutation()
        weightsA = mutant.weights
        biasesA = mutant.bias
    if random.random() < RATIO_MUTANTS:
        mutant = mutation()
        weightsB = mutant.weights
        biasesB = mutant.bias

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
        line = [mlpControl] + list(pendulum.rotational)
        controls.append(line)

    return controls

def testPopulation(population, pendulums):
    p = Pool(POOLS)
    args = []
    initialRotation = pendulums[0].rotational

    for x in range(ORGANISIMS):
        args.append((population[x], pendulums[x], ITERATIONS))
    Lpairs = p.map(testOrganism, args)

    sortedPairs = sortByFitness(Lpairs)
    print("initial [theta, omega]: " + str(list(initialRotation)) + ", sum abs(error): " + str(sortedPairs[0][0]))

    p.close()
    return map(lambda x : x[1], sortedPairs )

def populate():
    organisms = []
    for x in range(ORGANISIMS):
        org = MLP(2, NEURON_COUNT, FUNCTS)
        org.genWB(WEIGHT_RANGE)
        organisms.append(org)

    return organisms

def generatePendulums(rotational):
    pendulums = []
    for x in range(ORGANISIMS):
        pendulums.append(InvertedPendulum(rotational))

    return pendulums


if __name__ == "__main__":
    population = populate()
    best = None

    for x in range(EPOCH):
        initialPositions = array([pi + 0.05 * x % 1.45 - 0.70, 0])
        pendulums = generatePendulums(initialPositions)
        sortedPopulation = testPopulation(population, pendulums)
        if x == EPOCH - 1:
            best = sortedPopulation[0]

        population = crossBreed(sortedPopulation)

    newPend = InvertedPendulum()
    bestControls = outputControls(best,newPend,ITERATIONS)

    log = Logger("controls.csv")
    log.write(bestControls)
    log.close()

    print(best.weights)

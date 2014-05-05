import random
<<<<<<< Updated upstream
import sys

=======
import getopt
import sys
import argparse
>>>>>>> Stashed changes
from logger import *
from mlp import *
from pendulum import *
from numpy import *
from multiprocessing import Pool

random.seed()

ORGANISMS = 100
TOP = 15
RATIO_MUTANTS = 0.25
NEURON_COUNT = [2, 10, 10, 10, 10, 1]
FUNCTS = ["tansig", "tansig", "tansig", "tansig", "tansig", "purelin"]
ITERATIONS = 1500
WEIGHT_RANGE = 3.0
EPOCH = 1000
POOLS = 64
NUM_INPUTS = 4

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

    child = MLP(NUM_INPUTS, NEURON_COUNT, FUNCTS)
    child.weights = newWeights
    child.bias = newBiases

    return child

def crossBreed(L):
    nextGeneration = [L[0]]
    count = 0

    for i in range(TOP):
        parent1 = L[i]
        for j in range(i+1, TOP, 1):
            count += 1
            parent2 = L[j]

            nextGeneration.append(breed(parent1, parent2))

            if count == ORGANISMS:
                return nextGeneration

    return nextGeneration

def sortByFitness(Lpair):
    return sorted(Lpair, cmp = lambda x, y : 1 if x[0] > y[0] else -1)

def mutation():
    mutant = MLP(NUM_INPUTS,NEURON_COUNT, FUNCTS)
    mutant.genWB(WEIGHT_RANGE)
    return mutant

def testOrganism((mlp, pendulum, steps)):
    errs =[]

    for x in range(steps):
        mlpInputs = list(pendulum.rotational) + list(pendulum.translational)
        mlpInputs = array(map(lambda x: [x], mlpInputs))
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]

        pendulum.update(mlpControl)
        rotationError = abs(pendulum.rotational[0]- pi) + abs(pendulum.rotational[1])

        if abs(pendulum.rotational[0] - pi) > pi/2.0:
            rotationError = 1e9

        translationalError = abs(pendulum.translational[0]) + abs(pendulum.translational[1])

        if abs(pendulum.translational[0] > 1.5):
            translationalError = 1e4

        error = rotationError + translationalError
        errs.append(error)

    return [sum(errs), mlp, pendulum]

def outputControls(mlp, pendulum, steps):
    controls = []
    for x in range(steps):
        mlpInputs = list(pendulum.rotational) +  list(pendulum.translational)
        mlpInputs = array(map(lambda x: [x], mlpInputs))
        mlp.aups(mlpInputs)
        mlpControl = mlp.mlp_output()[0,0]
        line = [mlpControl] + list(pendulum.rotational) + list(pendulum.translational)
        pendulum.update(mlpControl)
        controls.append(line)

    return controls

def testPopulation(population, pendulums):
    p = Pool(POOLS)
    args = []
    initialRotation = pendulums[0].rotational

    for x in range(ORGANISMS):
        args.append((population[x], pendulums[x], ITERATIONS))
    Lpairs = p.map(testOrganism, args)

    sortedPairs = sortByFitness(Lpairs)

    print("initial [theta, omega]: " + str(map(lambda x : "{:.4f}".format(x), list(initialRotation))) + ", sum abs(error): " + str(sortedPairs[0][0]))
    print("final [theta, omega]: " + str(sortedPairs[0][2].rotational))
    print("final [x, x']: " + str(sortedPairs[0][2].translational))

    p.close()
    return map(lambda x : x[1], sortedPairs )

def populate():
    organisms = []
<<<<<<< Updated upstream
    for x in range(ORGANISIMS):
        org = MLP(NUM_INPUTS, NEURON_COUNT, FUNCTS)
=======
    for x in range(ORGANISMS):
        org = MLP(2, NEURON_COUNT, FUNCTS)
>>>>>>> Stashed changes
        org.genWB(WEIGHT_RANGE)
        organisms.append(org)

    return organisms

def generatePendulums(rotational, translational):
    pendulums = []
<<<<<<< Updated upstream
    for x in range(ORGANISIMS):
        pendulums.append(InvertedPendulum(rotational, translational))
=======
    for x in range(ORGANISMS):
        pendulums.append(InvertedPendulum(rotational))
>>>>>>> Stashed changes

    return pendulums

if __name__ == "__main__":
    population = populate()
    best = None

    for x in range(EPOCH):

        print("progress: " + "{:.3f}".format(100.0 * float(x)/float(EPOCH)) + "%")
        
        thetaChoices = [random.uniform(-0.7,-0.6), random.uniform(0.6, 0.7)]
        initialRotation = array([pi + random.choice(thetaChoices), random.uniform(-0.1,0.1)])
        initialTranslation = array([random.uniform(-0.1,0.1), random.uniform(-0.1,0.1)])
        pendulums = generatePendulums(initialRotation, initialTranslation)
        sortedPopulation = testPopulation(population, pendulums)
        if x == EPOCH - 1:
            best = sortedPopulation[0]

        population = crossBreed(sortedPopulation)


    newPend = InvertedPendulum(array([pi+0.5,0]))
    bestControls = outputControls(best,newPend,ITERATIONS)

    log = Logger("controls.csv")
    log.write(bestControls)
<<<<<<< Updated upstream
    log.close()

    weightsFile = open("weights.txt", 'w')

    for m in best.weights:
        mlist = str(m.tolist())
        mlist = mlist.replace('[','{')
        mlist = mlist.replace(']','}')
        weightsFile.write(mlist + '\n')

    weightsFile.close()

    biasesFile = open("biases.txt", 'w')

    for m in best.bias:
        mlist = str(m.tolist())
        mlist = mlist.replace('[','{')
        mlist = mlist.replace(']','}')
        biasesFile.write(mlist + '\n')

    biasesFile.close()
=======
    log.close()
>>>>>>> Stashed changes

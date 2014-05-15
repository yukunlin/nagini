import random
import sys
import multiprocessing

from mlp import *
from pendulum import *
from numpy import *

random.seed()

ORGANISIMS = 100
TOP = 15
RATIO_MUTANTS = 0.15
NEURON_COUNT = [4, 10, 10, 10, 1]
FUNCTS = ["purelin", "purelin", "purelin", "purelin", "purelin"]
ITERATIONS = 1000
WEIGHT_RANGE = 4.0
EPOCH = 500
POOLS = multiprocessing.cpu_count()
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

            if count == ORGANISIMS:
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
        rotationError = (pendulum.rotational[0]- pi)**2 + (pendulum.rotational[1])**2
        translationalError = (pendulum.translational[0])**2 + (pendulum.translational[1])**2

        if abs(pendulum.rotational[0] - pi) > pi/2.0:
            rotationError = 1e9

        if abs(pendulum.translational[0]) > 1.3:
            translationalError =  1e4

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

def testPopulation(population, pendulumLeft, pendulumRight):
    p = multiprocessing.Pool(POOLS)
    argsLeft = []
    argsRight = []
    pairs = []

    initialRotation = pendulumLeft[0].rotational

    for x in range(ORGANISIMS):
        argsLeft.append((population[x], pendulumLeft[x], ITERATIONS))

    for x in range(ORGANISIMS):
        argsRight.append((population[x], pendulumRight[x], ITERATIONS))

    Lpairs = p.map(testOrganism, argsLeft)
    Rpairs = p.map(testOrganism, argsRight)

    for x in range(ORGANISIMS):
        pairs.append( [ Lpairs[x][0] + Rpairs[x][0], Lpairs[x][1], Lpairs[x][2], Rpairs[x][2] ] )

    sortedPairs = sortByFitness(pairs)

    print("initial offset: " + str( list( abs( initialRotation - array([pi, 0]) ) ) ) )

    print("final left  rotational: " + str(sortedPairs[0][2].rotational))
    print("final right rotational: " + str(sortedPairs[0][3].rotational))

    print("final left  translational: " + str(sortedPairs[0][2].translational))
    print("final right translational: " + str(sortedPairs[0][3].translational))

    print("min error: " + str(sortedPairs[0][0]))

    p.close()
    return map(lambda x : x[1], sortedPairs )

def populate():
    organisms = []
    for x in range(ORGANISIMS):
        org = MLP(NUM_INPUTS, NEURON_COUNT, FUNCTS)
        org.genWB(WEIGHT_RANGE)
        organisms.append(org)

    return organisms

def generatePendulums(rotational, translational):
    pendulums = []
    for x in range(ORGANISIMS):
        pendulums.append(InvertedPendulum(rotational, translational))

    return pendulums


if __name__ == "__main__":
    population = populate()
    best = None

    for x in range(EPOCH):


        thetaChoices = [random.uniform(-0.7,-0.4), random.uniform(0.4, 0.7)]

        initialRotationLeft = array([pi + thetaChoices[0], random.uniform(-0.5,0.5)])
        initialRotationRight = array([pi + thetaChoices[1], random.uniform(-0.5,0.5)])
        initialTranslation = array([random.uniform(-0.1,0.1), random.uniform(-0.1,0.1)])

        pendulumLeft = generatePendulums(initialRotationLeft, initialTranslation)
        pendulumRight = generatePendulums(initialRotationRight, initialTranslation)

        sortedPopulation = testPopulation(population, pendulumLeft, pendulumRight)
        if x == EPOCH - 1:
            best = sortedPopulation[0]

        population = crossBreed(sortedPopulation)
        print("progress: " + "{:.3f}".format(100.0 * float(x)/float(EPOCH)) + "%")


    newPend = InvertedPendulum(array([pi+0.5,0]))
    bestControls = outputControls(best,newPend,ITERATIONS)

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
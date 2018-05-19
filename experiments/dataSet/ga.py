import numpy as np

class GA(object):

    def __init__(self):

    def loopOverParticles(self):
        for j in range(self.num_particles):
            print (j)


    def train(self, iterations, sampleSize):
        self.createParticles()

        trainingErrors = []

        for x in range(iterations):

            self.loopOverParticles()

            if (x % sampleSize == 0):
                trainingErrors.append([self.group_best_error, x])

        trainingErrors.append([self.group_best_error, iterations])

        return trainingErrors, self.group_best_error

import numpy as np

from mlpy.particleSwarmOptimization.pso import PSO
from src.MOParticle import MOParticle

class VEPSO(PSO):

    def __init__(self):
        super(VEPSO, self).__init__()

        self.group_best_objectives = None
        self.best_objectives = None

        self.group_best_error2 = float('inf')
        self.group_best_position2 = None
        self.group_best_objectives2 = None

        self.best_error2 = float('inf')
        self.best_position2 = None
        self.best_objectives2 = None

        self.swarm2 = []

        self.archiveSize = None
        self.archive = []

    def getGlobalBest(self):
        self.best_error = float('inf')
        for particle in self.swarm:
            if particle.best_error < self.group_best_error and particle.valid:
                self.group_best_position = np.array(particle.best_position)
                self.group_best_error = particle.best_error
                self.group_best_objectives = particle.best_objective
            # Get current best as well
            if particle.best_error < self.best_error and particle.valid:
                self.best_position = np.array(particle.best_position)
                self.best_error = particle.best_error
                self.best_objectives = particle.best_objective
        if self.group_best_position is None:
            self.group_best_position = np.array(self.swarm[0].best_position)
            self.group_best_error = self.swarm[0].best_error
            self.group_best_objectives = self.swarm[0].best_objective


        self.best_error2 = float('inf')
        for particle in self.swarm2:
            if particle.best_error < self.group_best_error2 and particle.valid:
                self.group_best_position2 = np.array(particle.best_position)
                self.group_best_error2 = particle.best_error
                self.group_best_objectives2 = particle.best_objective
            # Get current best as well
            if particle.best_error < self.best_error2 and particle.valid:
                self.best_position2 = np.array(particle.best_position)
                self.best_error2 = particle.best_error
                self.best_objectives2 = particle.best_objective
        if self.group_best_position2 is None:
            self.group_best_position2 = np.array(self.swarm2[0].best_position)
            self.group_best_error2 = self.swarm2[0].best_error
            self.group_best_objectives2 = self.swarm2[0].best_objective

        return self.group_best_position

    def createParticles(self):

        for i in range(self.num_particles):
            self.swarm.append(MOParticle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm[i].initPos(position, velocity)

            self.swarm2.append(MOParticle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm2[i].initPos(position, velocity)

    def loopOverParticles(self):
        for j in range(self.num_particles):
            objectives, valid = self.error.getFitness(self.swarm[j].position)
            self.swarm[j].error = objectives[0]
            self.swarm[j].objectives = objectives
            self.swarm[j].valid = valid
            if self.swarm[j].best_position is None:
                self.swarm[j].getPersonalBest()
                self.swarm[j].best_objective = objectives
            elif valid:
                old_objective, valid = self.error.getFitness(self.swarm[j].best_position)
                if self.error.dominant(objectives, old_objective):
                    self.swarm[j].getPersonalBest()
                    self.swarm[j].best_objective = objectives

            objectives, valid = self.error.getFitness(self.swarm2[j].position)
            self.swarm2[j].error = objectives[1]
            self.swarm2[j].objectives = objectives
            self.swarm2[j].valid = valid
            if self.swarm2[j].best_position is None:
                self.swarm2[j].getPersonalBest()
                self.swarm2[j].best_objective = objectives
            elif valid:
                old_objective, valid = self.error.getFitness(self.swarm2[j].best_position)
                if self.error.dominant(objectives, old_objective):
                    self.swarm2[j].getPersonalBest()
                    self.swarm2[j].best_objective = objectives

        self.getGlobalBest()

        for j in range(self.num_particles):
            self.swarm[j].update_velocity(self.group_best_position2)
            self.swarm[j].update_position()

            self.swarm2[j].update_velocity(self.group_best_position)
            self.swarm2[j].update_position()

            self.swarm[j].position = np.clip(self.swarm[j].position, self.bounds.minBound, self.bounds.maxBound)
            self.swarm2[j].position = np.clip(self.swarm2[j].position, self.bounds.minBound, self.bounds.maxBound)


    def train(self, iterations, sampleSize):
        self.createParticles()

        trainingErrors = []
        trainingErrors2 = []

        for x in range(iterations):

            self.loopOverParticles()

            if (x % sampleSize == 0):
                trainingErrors.append([self.group_best_objectives, x])
                trainingErrors2.append([self.group_best_objectives2, x])
                print(x, self.group_best_objectives, self.group_best_objectives2)

        trainingErrors.append([self.group_best_objectives, iterations])
        trainingErrors2.append([self.group_best_objectives2, iterations])

        return trainingErrors, trainingErrors2, self.group_best_objectives, self.group_best_objectives2

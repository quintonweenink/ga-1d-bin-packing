import numpy as np

from mlpy.particleSwarmOptimization.structure.particle import Particle

class MOParticle(Particle):
    def __init__(self, bounds, weight, cognitiveConstant, socialConstant):
        self.position = None  # particle position
        self.velocity = None  # particle velocity

        self.best_position = None  # best position individual
        self.best_error = float('inf')  # best error individual
        self.best_objectives = None
        self.valid = None

        self.error = None
        self.objectives = None

        self.num_dimensions = None
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant
        self.bounds = bounds

        self.neighbourhood = []

    def initPos(self, position, velocity):
        self.num_dimensions = len(position)

        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def getPersonalBest(self):
        if self.error < self.best_error:
            self.best_position = np.array(self.position)
            self.best_error = self.error

        return self.best_error

    def update_velocity(self, group_best_position):
        r1 = np.random.random(self.num_dimensions)
        r2 = np.random.random(self.num_dimensions)

        vel_cognitive = self.cognitiveConstant * r1 * (self.best_position - self.position)
        vel_social = self.socialConstant * r2 * (group_best_position - self.position)
        vel_inertia = self.weight * self.velocity
        self.velocity = vel_inertia + vel_cognitive + vel_social

        return self.velocity

    def update_position(self):
        self.position = self.position + self.velocity

        return self.velocity


    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self.position,
                                        pos_best=self.best_position,
                                        err=self.error)

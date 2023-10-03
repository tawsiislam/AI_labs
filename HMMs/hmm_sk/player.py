#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
from hmm3 import BaumWelch_Algo
from hmm1 import ForwardAlgorithm, VectorMultiplication


def row_stoch(list_len: int, biased_mean = 0):
    if biased_mean == 0: 
        mean = 1/list_len
    else:
        mean = biased_mean  # Option to choose the mean
    std = 0.1   # Standard deviation
    row = []
    row_sum = 0
    for elem in range(list_len):
        value = random.gauss(mean,std) #todo make sure to avoid negatives
        row.append(value)
        row_sum += value
    for elem in row:
        row[elem] /= row_sum    #Normalise the row for row stochastic rows
    return row


class HMM_model:
    def __init__(self):
        """
        Initialise row stochastic matrices where A is NxN, B is NxM and pi is 1xN.
        N = N_SPECIES
        M = N_EMISSIONS
        """
        self.pi = row_stoch(N_SPECIES)
        self.A = [row_stoch(N_SPECIES) for specie in range(N_SPECIES)]
        self.B = [row_stoch(N_EMISSIONS) for specie in range(N_SPECIES)]

    def updateModel(self,emissionSeq):
        """
        Update the model using Baum-Welch using previous model and emission sequence.
        """
        self.A, self.B, self.pi = BaumWelch_Algo(self.A, self.B, self.pi, emissionSeq)


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.fishModels = [HMM_model() for specie in range(N_SPECIES)]  #Initialise a model for each type of fish
        self.fishObs = [(fish_id,[]) for fish_id in range(N_FISH)]  #Create a tuple for each individual fish which has a list that will have all observations
        self.curr_fish_id = -1

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        # return (step % N_FISH, random.randint(0, N_SPECIES - 1))

        for fish_id in range(N_FISH):
            self.fishObs[fish_id][1].append(observations[fish_id])    #Add observations for all fish

        if step < 10: # Consider waiting for more observation
            return None #Make no guesses until we have acquired enough observations
        
        self.curr_fish_id += 1
        guessFishObs = self.fishObs[self.curr_fish_id][1]   #Extract observations for the individual fish we will guess
        bestModelProb = 0
        bestFishType = 0
        for fishType in range(N_SPECIES): #For each fish specie
            """Get the fish and it's observation
            For each specie, calculate which model has the highest probability for this particular fish with its observations
            """
            fishTypeModel = self.fishModels[fishType]
            alpha0 = VectorMultiplication(fishTypeModel.pi,fishTypeModel.B[guessFishObs[0]])    #Dot-product using initial probability (pi) with emission probability for fishType (B[fishType]) at t=0
            fishTypeProb = ForwardAlgorithm(alpha0,fishTypeModel.B,guessFishObs,fishTypeModel.A,len(fishTypeModel.A[0]),len(fishTypeModel.A))
            if fishTypeProb > bestModelProb:    #Saving which guess gave best probability
                bestModelProb = fishTypeProb
                bestFishType = fishType

        return (self.curr_fish_id, bestFishType)
        

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        if not correct:
           self.fishModels[true_type].updateModel(self.fishObs[fish_id])    # After knowing correct type, take its previous model and update it with Baum-Welch together with this particular fish_id's observation

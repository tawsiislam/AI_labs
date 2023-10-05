#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import sys
sys.path.append("../") #Should be commented out before running in Kattis
from hmm3 import BaumWelch_Algo
from hmm2 import ForwardAlgorithm

def row_stoch(list_len: int, biased_mean = 0):
    """ 
    Creates a normal distribution that is row stochastic (row sum to 1)
    """

    if biased_mean == 0: 
        mean = 1/list_len
    else:
        mean = biased_mean  # Option to choose the mean
    std = sys.float_info.epsilon   # Standard deviation
    row = []
    row_sum = 0
    for elem in range(list_len):
        value = abs(random.gauss(mean,std)) # using "abs" make sure to avoid negatives
        row.append(value)
        row_sum += value
    for elem in range(len(row)):
        row[elem] /= row_sum    #Normalise the row for row stochastic rows
    return row

class HMM_model:
    def __init__(self):
        """
        Initialise row stochastic matrices where A is NxN, B is NxM and pi is 1xN.
        N = N_SPECIES
        M = N_EMISSIONS
        """
        # lambda = (A,B,pi)
        self.pi = row_stoch(N_SPECIES) # initial probability for every species
        self.A = [row_stoch(N_SPECIES) for specie in range(N_SPECIES)] # every column is the current fish we believe; row is probability of changing our guess to another fish species (depending on the column)
        self.B = [row_stoch(N_EMISSIONS) for specie in range(N_SPECIES)] # given an observation (emission) whats the probability of it being that specie

    def updateModel(self,emissionSeq):
        """
        Update the model using Baum-Welch using previous model and emission sequence.
        Update is performed when we guess wrong species.
        """
        self.A, self.B, self.pi = BaumWelch_Algo(self.A, self.B, self.pi, emissionSeq,50) # 50 iterations of Baum-Welch


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.fishModels = [HMM_model() for specie in range(N_SPECIES)]  #Initialise a model for each type of fish
        self.fishObs = [(fish_id,[]) for fish_id in range(N_FISH)]  #Create a tuple for each individual fish which has a list that will have all observations
        self.curr_fish_id = -1 # to start with fish id 0 when guessing

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

        # if step == 1:
        #     import debugpy
        #     debugpy.listen(50452)
        #     print("Attach debugger please")
        #     debugpy.wait_for_client()
        #     print("Debugger attached")

        if step < (N_STEPS - N_FISH): # Consider waiting for more observation. Waiting for long as possible
            return None #Make no guesses until we have acquired enough observations
        
        self.curr_fish_id += 1
        guessFishObs = self.fishObs[self.curr_fish_id][1]   #Extract observations for the individual fish we will guess
        
        # intialise best guess
        bestModelProb = 0
        bestFishType = 0
        for fishType in range(N_SPECIES): #For each fish specie, one model per species
            """Get the fish and it's observation
            For each specie, calculate which model has the highest probability for this particular fish with its observations
            """
            fishTypeModel = self.fishModels[fishType]

            fishTypeProb = ForwardAlgorithm(fishTypeModel.pi,fishTypeModel.B,guessFishObs,fishTypeModel.A,len(fishTypeModel.A[0]),len(fishTypeModel.A),returnSum=False)

            fishTypeProb = fishTypeProb[-1] # only interested in probability last observation
            
            if fishTypeProb > bestModelProb:    #Saving which guess gave best probability
                bestModelProb = fishTypeProb
                bestFishType = fishType

        return (self.curr_fish_id, bestFishType) # returning guess of fish_id and fish_type
        

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
           # After knowing correct type, take its previous model and update it 
           # with Baum-Welch together with this particular fish_id's observation
           self.fishModels[true_type].updateModel(self.fishObs[fish_id][1])    
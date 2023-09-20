#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from typing import List, Tuple # for type hinting
import numpy as np
import time


"""
improvements:
- don't need to evaluate stay action every time?
"""



class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        
        # initializing variables for time/depth limit for minimax
        # self.timeLimit = 0.075 # time limit for each move in seconds, 75 ms
        self.timeLimit = 0.075*0.8
        # self.max_depth = 3 # 5 is good depth
        self.max_depth = 7
        
        self.maxAchievedDepth = 0
        super(PlayerControllerMinimax, self).__init__()
        

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """
        start_time = time.time()

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()
        
        

        self.statesHashTable = dict()

        while True:
            msg = self.receiver()

            
            """ 
            message is a dictionary with the following keys:
            dict_keys(['game_over', 'hooks_positions', 'fishes_positions', 'observations', 'fish_scores', 'player_scores', 'caught_fish'])

            'game_over' is a boolean indicating if the game is over
            
            'hooks_positions' is a dictionary with keys 0 and 1, each containing a tuple with the position of the hook of each player

            'fishes_positions' dictionary of tuples with the position of each fish

            'observations' ?

            'fish_scores'  same dictionary keys as 'fishes_positions' but with the score of each corresponding fish

            'player_scores' is a dictionary with keys 0 and 1, each containing the score of each player

            'caught_fish' is a dictionary with keys of corresponding fish that players are currently holding on hook, can check the fishes score value by using the key to index into 'fish_scores'

            """


            # Create the root node of the game tree
            node = Node(message=msg, player=0)
            
            
            


            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def HeuristicFunc(self, parentNode, player): #Needs to modified to be own. This is copied
        max_hook = parentNode.state.get_hook_positions()[0] 
        fishes = parentNode.state.get_fish_positions()
        fish_scores = parentNode.state.get_fish_scores()
        gameScore = parentNode.state.get_player_scores()[0] - parentNode.state.get_player_scores()[1]
        total_heur = 0 
        
        for fish_num, fish_pos in fishes.items():

            x = fish_pos[0]
            y = fish_pos[1]

            distance_from_max_hook = abs(max_hook[0] - x) + abs(max_hook[1] - y)

            total_heur += (fish_scores[fish_num]) / (distance_from_max_hook + 0.01) 
        return total_heur + gameScore*10    #TODO improve heuristic
    
    def hashkey(self, parentNode):
        hookPos = parentNode.state.get_hook_positions()
        fishDict = dict()
        for fishScore, fishPos in zip(parentNode.state.get_fish_scores().items(), parentNode.state.get_fish_positions().items()):
            fishKey = fishPos
            fishDict.update({fishKey:fishScore})
        return str(hookPos)+str(fishDict)



    def alphabeta(self, parentNode:Node, depth:int, visitedStates:dict, initialTime:int, alpha:int, beta:int, player:int) -> int:
        timeExceeded = False
        stateKey = self.hashkey(parentNode)
        
        nodeChildren = parentNode.compute_and_get_children()


        # for each child, calculate an initial heuristic value, and sort the children based on this: move ordering

        heuristicValuesChildren = []
        for childNo,child in enumerate(nodeChildren):
            heuristicValueChild = self.HeuristicFunc(child, player)
            heuristicValuesChildren.append(heuristicValueChild)

        # performing the sorting
        sortedIndices = np.argsort(heuristicValuesChildren) #Move ordering
        nodeChildren = np.array(nodeChildren)[sortedIndices]


        # check if depth limit reached or if time is exceeded        
        if depth == 0 or time.time()-initialTime >= self.timeLimit:
            bestPossibleValue = self.HeuristicFunc(parentNode, player)
            # if time.time()-initialTime >= self.timeLimit: timeExceeded  = True
            return bestPossibleValue, timeExceeded
        
        # ----otherwire continue search----

        # check if state has been visited before
        if stateKey in visitedStates:
            return visitedStates[stateKey], timeExceeded

        # unvisited state: check if max or min player

        elif player == 0:   #max player
            bestPossibleValue = -np.inf
            for child in nodeChildren:
                childBestValue, timeExceeded = self.alphabeta(child, depth-1, visitedStates, initialTime, alpha, beta, 1)
                bestPossibleValue = np.max([bestPossibleValue, childBestValue]) 
                # if timeExceeded == True:
                    # return bestPossibleValue, timeExceeded
                alpha = np.max([alpha, bestPossibleValue])

                # update value of state in hash table
                visitedStates.update({stateKey:bestPossibleValue})
                if beta <= alpha:
                    break
        else:
            bestPossibleValue = np.inf
            
            for child in nodeChildren:
                childBestValue, timeExceeded = self.alphabeta(child, depth-1, visitedStates, initialTime, alpha, beta, 0)
                bestPossibleValue = np.min([bestPossibleValue, childBestValue])
                # if timeExceeded == True:
                    # return bestPossibleValue, timeExceeded
                beta = np.min([beta, bestPossibleValue])

                # update value of state in hash table
                visitedStates.update({stateKey:bestPossibleValue})
                if beta <= alpha:
                    break
        return bestPossibleValue, timeExceeded
    

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!


        # start time to keep track of time limit
        startTime = time.time()
        depth = 0
        bestOverallHeur = -np.inf
        # bestMoveInt = 0 # 0 is stay, 1 is left, 2 is right, 3 is up, 4 is down

        # dictionary to keep track of visited nodes
        visitedStates = dict()
        # change
        


        # get children of the initial tree node
        children = initial_tree_node.compute_and_get_children()
        # for depth in range (1,4):   #IDDFS
        depth = 1
        while (time.time()-startTime < self.timeLimit): #IDDFS
            
            heuristicValuesChildren = []
            for childNo,child in enumerate(children):
                heuristicValueChild, timeExceeded = self.alphabeta(child, depth-1, visitedStates, startTime, -np.inf, np.inf, 0)
                heuristicValuesChildren.append(heuristicValueChild)
                # if timeExceeded == True: break
            bestChildHeur = np.max(heuristicValuesChildren)
            if bestChildHeur > bestOverallHeur: 
                bestOverallHeur = bestChildHeur
                bestMove = ACTION_TO_STR[children[np.argmax(heuristicValuesChildren)].move]

            
            #Move ordering
            sortedIndices = np.argsort(heuristicValuesChildren) 
            children = np.array(children)[sortedIndices]
            
            # check if time is exceeded
            # if timeExceeded == True: break
            # bestMove = ACTION_TO_STR[children[bestMoveInt].move]
            
            depth += 1



            if depth > self.maxAchievedDepth:
                self.maxAchievedDepth = depth
                print("max achieved depth: ", self.maxAchievedDepth)

        # random_move = random.randrange(5)
        return bestMove
        
# convert for to while, stop when time is exceeded

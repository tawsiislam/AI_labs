#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from typing import List, Tuple # for type hinting
import numpy as np
import time

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
        super(PlayerControllerMinimax, self).__init__()
        

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()
        
        # initializing variables for time/depth limit for minimax
        start_time = time.time()
        max_depth = 3

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
        bestMoveInt = 0 # 0 is stay, 1 is left, 2 is right, 3 is up, 4 is down

        # dictionary to keep track of visited nodes
        visitedNodes = dict()


        
        
        

        random_move = random.randrange(5)
        return ACTION_TO_STR[random_move]





    def DepthSearch(rootNode:Node, depth:int, maxDepth:int, initialTime:int, maxTime:int, visitedNodes) -> Node:
        """depth first search algorithm, using alpha beta pruning"""

        # check if the state is terminal
        if CheckIfTerminalState(rootNode.state, depth, maxDepth, time.time() - initialTime, maxTime):
            return rootNode
        

        # initial values for alpha and beta
        alpha = -np.inf
        beta = np.inf



        # compute children of the node
        children = rootNode.compute_and_get_children()

        # each child is to be assigned a heuristic value, they are saved in a list, so can choose the best one
        heuristicValuesChildren = []

        # loop through children
        for child in children:
                
            # check if child has been visited
            if child in visitedNodes:
                continue
            
            # add child to visited nodes
            visitedNodes[child] = True

              
                

        return Node()
        

    def AlphaBetaPruning(rootNode:Node, depth:int, maxDepth:int, initialTime:int, maxTime:int, alpha:int, beta:int) -> int:
        """alpha beta pruning algorithm for minimax; 
            returns the heuristic value of the node
        """

        # check if the state is terminal, raise exception
        if CheckIfTerminalState(rootNode.state, depth, maxDepth, time.time() - initialTime, maxTime):
            return RuntimeError("Error: reached time limit")
        



        
        return 0



    def HeuristicFunction(state:State, indxClosestFish) -> int:
        """heuristic function for the minimax algorithm"""
        
        value = state.player_scores[0] - state.player_scores[1]


        return 0






# distance calculator, euclidian, takes 2 tuples and returns distance between them
def ManhattanDistance(coord1: Tuple, coord2:Tuple)->float:
    """takes in two coordinates in form of tuples and returns the manhattan distance between them
        is |dx|+|dy|
    """

    # return np.sqrt( (coord2(1) -coord1(1))**2 + (coord2(2) -coord1(2))**2  )

    # when being on the edge of the map one can make a step that gets you to the other side of the map
    dx = np.min([np.abs(coord1[0] - coord2[0]), 20 - np.abs(coord1[0] - coord2[0])])
    
    # for y it is normal distance calculation
    dy = np.abs(coord1[1] - coord2[1])

    return dx + dy


class ParentNode:
    def __init__(self, currPoint:tuple, parent, heurVal:int = 0 ) -> None:
        self.currentNode = currPoint
        self.parent = parent
        self.heuristicValue = heurVal
        

    def SetHeuristicValue(self, heusVal:int) ->None:
        self.heuristicValue = heusVal
        
        
def CheckIfTerminalState(state, depth, maxDepth, time, maxTime) -> bool:
    """checks if the state is terminal or not, returns true if terminal, false if not"""
    if depth == maxDepth or time == maxTime: # or state.game_over == True
        return True
    else:
        return False


# evaluation function
#- heuristic function




# minimax function
#- max function
#- min function


# alpha beta pruning function









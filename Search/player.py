#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from typing import List, Tuple # for type hinting
import numpy as np

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

        
        

        random_move = random.randrange(5)
        return ACTION_TO_STR[random_move]





# distance calculator, euclidian, takes 2 tuples and returns distance between them
def EuclidianDist(coord1: Tuple, coord2:Tuple)->float:
    """takes in two coordinates in form of tuples and returns distance between them"""

    return np.sqrt( (coord2(1) -coord1(1))**2 + (coord2(2) -coord1(2))**2  )


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









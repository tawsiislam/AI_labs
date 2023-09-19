#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
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
        self.timeLimit = 0.075*0.8
        self.max_depth = 3
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

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def HeuristicFunc(self, state, player): #Needs to modified to be own. This is copied
        max_hook = state.state.get_hook_positions()[0] 
        fishes = state.state.get_fish_positions()
        fish_scores = state.state.get_fish_scores()
        total_utility = 0 
        
        for fish_num, fish_pos in fishes.items():

            x = fish_pos[0]
            y = fish_pos[1]

            distance_from_max_hook = abs(max_hook[0] - x) + abs(max_hook[1] - y)

            total_utility += (fish_scores[fish_num]) / ((distance_from_max_hook + 0.01) ** 2) 
        return total_utility

    def alphabeta(self, parentNode:Node, depth:int, initialTime:int, alpha:int, beta:int, player:int) -> int:
        nodeChildren = parentNode.compute_and_get_children()
        if depth == 0 or time.time()-initialTime == self.timeLimit:
            bestPossibleValue = self.HeuristicFunc(parentNode, player)

        elif player == 0:   #max player
            bestPossibleValue = -np.inf
            for child in nodeChildren:
                bestPossibleValue = np.max([bestPossibleValue, self.alphabeta(child, depth-1, initialTime, alpha, beta, 1)]) 
                alpha = np.max([alpha, bestPossibleValue])
                if beta <= alpha:
                    break
        else:
            bestPossibleValue = np.inf
            
            for child in nodeChildren:
                bestPossibleValue = np.min([bestPossibleValue, self.alphabeta(child, depth-1, initialTime, alpha, beta, 0)])
                beta = np.min([beta, self.alphabeta(child, depth-1, initialTime, alpha, beta, 1)])
                if beta <= alpha:
                    break
        return bestPossibleValue
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

        startTime = time.time()
        depth = self.max_depth
        children = initial_tree_node.compute_and_get_children()
        heuristicValuesChildren = []
        for child in children:
            heuristicValuesChildren.append(self.alphabeta(child, depth-1, startTime, -np.inf, np.inf, 0))
        bestMoveInt = np.argmax(heuristicValuesChildren)
        bestMove = ACTION_TO_STR[children[bestMoveInt].move]

        # random_move = random.randrange(5)
        return bestMove

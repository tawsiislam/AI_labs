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

    def HeuristicFunc(self, parentNode, player): #Needs to modified to be own. This is copied
        max_hook = parentNode.state.get_hook_positions()[0] 
        min_hook = parentNode.state.get_hook_positions()[1]
        fishes = parentNode.state.get_fish_positions()
        fish_scores = parentNode.state.get_fish_scores()
        gameScore = parentNode.state.get_player_scores()[0] - parentNode.state.get_player_scores()[1]
        maxHeur = 0
        minHeur = 0 
        
        for fish_num, fish_pos in fishes.items():
            maxDist = self.ManhattanDistance(max_hook,fish_pos)
            minDist = self.ManhattanDistance(min_hook,fish_pos)

            if maxDist == 0:
                maxHeur = fish_scores[fish_num] * 2
            else: 
                maxHeur += (fish_scores[fish_num]) / (maxDist + 1) 
            if minDist == 0:
                minHeur = fish_scores[fish_num]
            else:
                minHeur += (fish_scores[fish_num]) / (minDist + 1) 
        return (maxHeur-minHeur) + gameScore*10    #TODO improve heuristic

    def ManhattanDistance(self, hookCoord: tuple, fishCoord:tuple)->float:
        """takes in two coordinates in form of tuples and returns the manhattan distance between them
            is |dx|+|dy|
        """

        # when being on the edge of the map one can make a step that gets you to the other side of the map
        dx = np.min([np.abs(hookCoord[0] - fishCoord[0]), 20 - np.abs(hookCoord[0] - fishCoord[0])])
        
        # for y it is normal distance calculation
        dy = np.abs(hookCoord[1] - fishCoord[1])

        return dx + dy
    
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
        if depth == 0 or time.time()-initialTime >= self.timeLimit:
            bestPossibleValue = self.HeuristicFunc(parentNode, player)
            # if time.time()-initialTime >= self.timeLimit: timeExceeded  = True
            return bestPossibleValue, timeExceeded
        
        elif stateKey in visitedStates:
            return visitedStates[stateKey], timeExceeded

        elif player == 0:   #max player
            bestPossibleValue = -np.inf
            for child in nodeChildren:
                childBestValue, timeExceeded = self.alphabeta(child, depth-1, visitedStates, initialTime, alpha, beta, 1)
                bestPossibleValue = np.max([bestPossibleValue, childBestValue]) 
                # if timeExceeded == True:
                    # return bestPossibleValue, timeExceeded
                alpha = np.max([alpha, bestPossibleValue])
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

        startTime = time.time()
        bestOverallHeur = -np.inf
        timeExceeded = False    #Was used to control if IDDFS would take too long, maybe better without
        visitedStates = dict()
        bestMove = ""
        children = initial_tree_node.compute_and_get_children()
        for depth in range (1,10):   #IDDFS
            depth = depth
            heuristicValuesChildren = []
            for childNo,child in enumerate(children):
                heuristicValueChild, timeExceeded = self.alphabeta(child, depth-1, visitedStates, startTime, -np.inf, np.inf, 0)
                heuristicValuesChildren.append(heuristicValueChild)
                # if timeExceeded == True: break
            bestChildHeur = np.max(heuristicValuesChildren)
            if bestChildHeur > bestOverallHeur: 
                bestOverallHeur = bestChildHeur
                bestMove = ACTION_TO_STR[children[np.argmax(heuristicValuesChildren)].move]

            sortedIndices = np.argsort(heuristicValuesChildren) #Move ordering
            children = np.array(children)[sortedIndices]
            # if timeExceeded == True: break
            # bestMove = ACTION_TO_STR[children[bestMoveInt].move]

        # random_move = random.randrange(5)
        return bestMove

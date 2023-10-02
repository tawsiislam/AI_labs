# HMM2 Estimate Sequence of States

# You will be given three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row)

# You should output the most probable sequence of states as zero-based indices separated by spaces. Do not output the length of the sequence.

from math import log



def ParseMatrix(matrixStr: str):
    """
    splits string matrix into a matrix of floats
    first 2 elements in each row are the dimensions of the matrix
    """
    nDim = int(matrixStr[0])
    mDim = int(matrixStr[1])

    matrix = []

    # makes a list of the matrix elements
    matrixElements = list(map(float, matrixStr[2:])) 


    for rowIndx in range(nDim):
        row = []
        for colIndx in range(mDim):

            # add element to row: rowIndx: is the row number, mDim: is the number of columns, colIndx: is the column number
            row.append(matrixElements[rowIndx * mDim + colIndx])
        matrix.append(row)
    return matrix
    

def matrix_multiplication(matA, matB):
    """
    Takes in two matrices and returns their product
    A: nDimA x mDimA
    B: nDimB x mDimB
    product: nDimA x mDimB

    requires mDimA == nDimB
    """
    # Get dimensions of the matrices
    nDimA = len(matA)
    mDimA = len(matA[0])

    nDimB = len(matB)
    mDimB = len(matB[0])

    # Check if matrices can be multiplied
    if mDimA != nDimB:
        raise Exception("Matrices cannot be multiplied")

    # Perform matrix multiplication using list comprehensions
    product = [[sum( matA[row][k] * matB[k][col] for k in range(mDimA) ) for col in range(mDimB)] for row in range(nDimA)]

    return product


def VectorMultiplication(vectorA: list, vectorB: list):
    """
    multiplies two vectors
    """
        
    # if len(vectorA) != len(vectorB):
    #     # if one is scalar and other is vector, multiply each element of vector by scalar
    #     if len(vectorA) == 1:
    #         return [vectorA[0] * b for b in vectorB]
    #     elif len(vectorB) == 1:
    #         return [a * vectorB[0] for a in vectorA]
        # else:
        #     raise Exception("Vectors must be of same length")

    return [a * b for a, b in zip(vectorA, vectorB)]


def GetCollumn(matrix: list, col: int):
    """
    returns a collumn from a matrix 
    """
    column = [row[col] for row in matrix]
    return column


def ForwardAlgorithm(alpha:float, emissionMatrix:list, observationsequence:list, transitionmatrix:list, j:int, N:int, returnSum:bool = True):
    """
    a.k.a. alpha-pass.
    Given an observation sequence, calculate the probability of the observation sequence
    for 2<=t<=T, i.e. initial alpha is given
    """
    if len(observationsequence) == 0:
        if returnSum:
            return sum(alpha) 
        else:
            return alpha
    
    # alphaNew is sum from 1 to N of alpha(i,t-1) * a(i,j) * b(j,ot)
    alphaNewTemp = [ sum(  VectorMultiplication( alpha , GetCollumn(transitionmatrix, i) )  ) for i in range(j) ]

    alphaNew = VectorMultiplication( alphaNewTemp, GetCollumn(emissionMatrix, observationsequence[0]) )
    
    # alphaNew = ForwardAlgorithm(alphaNew, emissionMatrix, observationsequence[1:], transitionmatrix, emissionmatrix, j, N)

    return ForwardAlgorithm(alphaNew, emissionMatrix, observationsequence[1:], transitionmatrix, j, N, returnSum)



# def ViterbiAlgorithm(delta:float, emissionMatrix:list, observationsequence:list, transitionmatrix:list, j:int, N:int, stateSequence:list = [], sumOfProbabilities:list = []):
    """
    a.k.a. Viterbi-pass.
    Given an observation sequence, calculate the most probable sequence of states
    for 1<t<=T,
    """

    if len(observationsequence) == 0:
        # sumOfProbabilities = [sum(x) for x in stateSequence]
        
        stateTransitionSequence = []

        lastIndx = delta.index(max(delta))
        stateTransitionSequence.append(lastIndx)

        for i in range(len(stateSequence)-1, 0, -1): # loop from last to second last element
            lastIndx = stateSequence[i][lastIndx]
            stateTransitionSequence.append(lastIndx)

        stateTransitionSequence.reverse()
        print(" ".join([str(x) for x in stateTransitionSequence]))


        return stateSequence, sumOfProbabilities, stateSequence.index(max(sumOfProbabilities))


    # delta : delta_{t-1} 
    # probabilities = [   [ log( delta[currState] * transitionmatrix[currState][nextState] *emissionMatrix[nextState][observationsequence[0]] ) for currState in range(j)] for nextState in range(j)   ] # todo: check if indexing is correct

    # given the current distibution (delta) calc. probability to transition to next state of the observation sequence
    probabilities = [ [ [] for i1 in range(j)] for i2 in range(j) ]
    
    # for currState in range(j):
    #     for nextState in range(j):
            
            
    #         # probTemp = transitionmatrix[currState][nextState] *emissionMatrix[nextState][observationsequence[0]] 
            
    #         # should be no scenario in which probTemp < 0
    #         if probTemp > 0:
    #             probTemp = log(probTemp) + delta[currState]
    #         else:
    #             probTemp = delta[currState]


    #         # note, since we use log space, delta can be negative
    #         probabilities[currState][nextState] = probTemp

    # delta_t = max(   [ [ delta[currState] + log( transitionmatrix[currState][nextState] * emissionMatrix[nextState][observationsequence[0]] ) for currState in range(j)] for nextState in range(j) ]   , key = max  )




    print("probabilities", probabilities)

    # having all probabilities calculated, we send the maximum probability to the next iteration

    maxProbability = [max(x) for x in probabilities] #delta_{t}


    stateSequence = [x.index(max(x)) for x in probabilities] 
    print("stateSequence", stateSequence)

    




    ViterbiAlgorithm(maxProbability, emissionMatrix, observationsequence[1:], transitionmatrix, j, N, stateSequence, sumOfProbabilities)

    return [], [], 0

    

def ViterbiAlgorithm(delta0:list, emissionMatrix:list, emissionSequence:list, transitionmatrix:list, j:int, N:int, stateSequence:list = [], sumOfProbabilities:list = []):
    
    pathProb = [[0 for emission in range(len(emissionSequence))] for state in range(j)] # matrix of size j x len(emissionSequence)
    stateProb = [[0 for emission in range(len(emissionSequence))] for state in range(j)]

    for state_i in range(len(delta0)):
        pathProb[0][state_i] = delta0[state_i]

    for obs_t in range(1, len(emissionSequence)):
        ptr = []
        for state_i in range(len(transitionmatrix)):
            deltaList = []
            for state_j in range(len(transitionmatrix)):
                deltaList.append(pathProb[obs_t-1][state_j] * transitionmatrix[state_j][state_i]* emissionMatrix[state_i][emissionSequence[obs_t]]) 
            max_path_prob =  max(deltaList) 
            pathProb[obs_t][state_i] = max_path_prob
            stateProb[obs_t][state_i] = deltaList.index(max_path_prob)

    # TODO: Traverse to find the paths

    return 0

def main():
    # read the inputs:
    A = [float(x) for x in input().split()] # transition matrix
    B = [float(x) for x in input().split()] # emission matrix
    pi = [float(x) for x in input().split()] # initial state probability distribution
    
    # there are M different emissions
    # for example if M = 3 possible different emissions, they would be identified by 0, 1 and 2 in the emission sequence
    emissionSequence = [int(x) for x in input().split()] # observation sequence
    
    # check number of different emmisions: # N is number of possible states
    M = max(emissionSequence[1:]) + 1 # +1 because emissionSequence starts at 0
    

    
    # convert inputs to matrices
    A = ParseMatrix(A)
    B = ParseMatrix(B)
    pi = ParseMatrix(pi)

    #---------------------------------------------------------------------------
    

    # j is number of collumns in A
    j = len(A[0])

    
    # emissionSequence[2:] because first element tells size, second element is first emission, etc
    
    
    # we do viterbi algorithm to find the most probable sequence of states
    
    

    # initialize delta
    delta0 = VectorMultiplication(pi[0], GetCollumn(B, emissionSequence[1] ) )

    # initialize deltaList as matrix of size len(emissionSequence) x j
    deltaList = [ [0 for i in range( len(delta0) )] for i in range(j) ]
    
    

    #deltaList[0] = delta0 # set first row of deltaList to delta0

    
    
    # convert to log space if not 0 or negative, this is to avoid underflow
    # delta0 = [log(x) if x > 0 else x for x in delta0]

    stateSequenceList = [[] for i in range(j)] # list of lists, each list is a state sequence
    sumOfProbabilities = [0 for i in range(j)] # list of probabilities for each state sequence
    indxMostProbable = 0 # index of most probable state sequence

    # print("stateSequenceList", stateSequenceList)
    # print("sumOfProbabilities", sumOfProbabilities)


    stateSequenceList, sumOfProbabilities, indxMostProbable = ViterbiAlgorithm(delta0, B, emissionSequence[1:], A, j, M, stateSequenceList, sumOfProbabilities)

    


    
    


"""
4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
4 1 1 2 2 


out: 0 1 2 1 

"""

    
    
    




if __name__ == "__main__":
    main()

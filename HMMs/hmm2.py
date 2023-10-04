# HMM2 Estimate Sequence of States

# You will be given three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row)

# You should output the most probable sequence of states as zero-based indices separated by spaces. Do not output the length of the sequence.





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






def ViterbiAlgorithm(delta0:list, emissionMatrix:list, emissionSequence:list, transitionmatrix:list, j:int, M:int, stateSequence:list = [], sumOfProbabilities:list = []):
    # M number of differnet emissions
    # j number of states


    

    # T1 is probability of most probable path so far
    T1 = [[0 for emission in range(len(emissionSequence))] for state in range(len(emissionMatrix[0]))] # matrix of size j x len(emissionSequence)
    
    # most likely path so far
    T2 = [[0 for emission in range(len(emissionSequence))] for state in range(len(emissionMatrix[0]))] # matrix of size j x len(emissionSequence)

    # stateProb = [[0 for emission in range(len(emissionSequence))] for state in range(j)]

    # save delta0 to T1
    for stateI in range(j):
        T1[stateI][0] = delta0[stateI]
        T2[stateI][0] = 0


    for observationJ in range(1, len(emissionSequence)):
        for stateI in range(j):
            
            temp = [ T1[stateK][observationJ-1] * transitionmatrix[stateK][stateI] * emissionMatrix[stateI][emissionSequence[observationJ]] for stateK in range(j) ]

            T1[stateI][observationJ] =  max( temp ) 

            T2[stateI][observationJ] = temp.index( T1[stateI][observationJ ])

    # find most probable state to transition to the last state
    
    stateSequence = [0 for i in range(len(emissionSequence))]
    
    # find most probable last state
    for stateI in range(j):
        stateSequence[-1] = stateI if T1[stateI][-1] > T1[stateSequence[-1]][-1] else stateSequence[-1]


    # traverse backwards
    for observationJ in range(len(emissionSequence)-2, -1, -1): # start: len(emissionSequence)-2, stop: 0, step: backwards
        stateSequence[observationJ] = T2[stateSequence[observationJ+1]][observationJ+1]
    

    print(  " ".join(map(str, stateSequence))   )

    return stateSequence






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

    
    
    # we do viterbi algorithm to find the most probable sequence of states
    
    

    # initialize delta
    delta0 = VectorMultiplication(pi[0], GetCollumn(B, emissionSequence[1] ) )


    stateSequenceList = [[] for i in range(j)] # list of lists, each list is a state sequence
    sumOfProbabilities = [0 for i in range(j)] # list of probabilities for each state sequence
    


    # emissionSequence[2:] because first element tells size, second element is first emission, etc
    stateSequenceList = ViterbiAlgorithm(delta0, B, emissionSequence[1:], A, j, M, stateSequenceList, sumOfProbabilities)

    


    
    


"""

4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
4 1 1 2 2 




5 5 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
5 5 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
1 5 1.0 0.0 0.0 0.0 1.0
5 1 1 2 2 1


out: 0 1 2 1 

"""

    
    
    




if __name__ == "__main__":
    main()

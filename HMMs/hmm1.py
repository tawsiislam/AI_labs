# HMM1 Probability of Emission Sequence

# You will be given three matrices (in this order); transition matrix, emission matrix, and initial state probability distribution. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input.

# output: calculate the probability for this sequence.




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

    # alpha is a list of probabilities for each state | return when no more observations
    if len(observationsequence) == 0:
        # return sum over all alpha = probability of observation sequence
        if returnSum:
            return sum(alpha) 
        else: 
            return alpha
    
    # alphaNew is sum from 1 to N of alpha(i,t-1) * a(i,j) * b(j,ot)
    alphaNewTemp = [ sum(  VectorMultiplication( alpha , GetCollumn(transitionmatrix, i) )  ) for i in range(j) ]

    alphaNew = VectorMultiplication( alphaNewTemp, GetCollumn(emissionMatrix, observationsequence[0]) )
    

    return ForwardAlgorithm(alphaNew, emissionMatrix, observationsequence[1:], transitionmatrix, j, N, returnSum)




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

    # calculate probability of emission sequence using forward algorithm
    # initial alpha,      pi[0] gives initial state probability distribution as a vector (list)
    aplpha1 = VectorMultiplication(pi[0], GetCollumn(B, emissionSequence[1] ) )
    

    
    
    # j is number of collumns in A
    j = len(A[0])

    

    # emissionSequence[2:] because first element tells size, second element is first emission, etc

   
    pObservationsGivenLambda = ForwardAlgorithm(aplpha1, B, emissionSequence[2:], A, j, M)
    
    
    print(pObservationsGivenLambda)
    




if __name__ == "__main__":
    main()

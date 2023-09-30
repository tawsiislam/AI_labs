import sys
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




def GetColumn(matrix: list, col: int):
    """
    returns a collumn from a matrix 
    """
    column = [row[col] for row in matrix]
    return column

def alphaPass(A: list, B: list, pi: list, O: list):
    totStateIterable = range(len(A)) #Substitue repeating range(len(A)) to create a loop
    totObsIterable = range(len(B))

    alpha = [[0 for obs in totObsIterable] for state in totStateIterable]
    c = [0 for obs in totObsIterable]
    for state in totStateIterable:
        alpha[0][state] = pi[state]*B[state][O[0]]
        c[0] += alpha[0][state]
    for state in totStateIterable:
        alpha[0][state] /= c[0]  #Normalising alpha[state][0]/c[0]

    for obs_t in totObsIterable[1:]:
        for state_i in totStateIterable:
            for state_j in totStateIterable:
                alpha[obs_t][state_i] = alpha[obs_t][state_i] + alpha[obs_t-1][state_j]*A[state_j][state_i]
            alpha[obs_t][state_i] *= B[state_i][O[obs_t]]
            c[obs_t] += alpha[obs_t][state_i]
        for state_i in totStateIterable:
            alpha[obs_t][state_i] /= c[obs_t]

    return alpha, c 



    

def BaumWelch_Algo(A: list, B: list, pi: list, O: list):
    max_iter = 100
    iter = 0
    overallLogProb = float(-'inf')


    pass

def main():
    print("run")
    # read the inputs:
    A = [float(x) for x in sys.argv[1].split()] # transition matrix
    B = [float(x) for x in sys.argv[2].split()] # emission matrix
    pi = [float(x) for x in sys.argv[3].split()] # initial state probability distribution
    
    # there are M different emissions
    # for example if M = 3 possible different emissions, they would be identified by 0, 1 and 2 in the emission sequence
    emissionSequence = [int(x) for x in sys.argv[4].split()] # observation sequence
    
    # check number of different emmisions: # N is number of possible states
    # M = max(emissionSequence[1:]) + 1 # +1 because emissionSequence starts at 0
    

    
    # convert inputs to matrices
    A = ParseMatrix(A)
    B = ParseMatrix(B)
    pi = ParseMatrix(pi)
    # emissionSequence = ParseMatrix(emissionSequence)


    #---------------------------------------------------------------------------
    alpha,c = alphaPass(A, B, pi[0], emissionSequence[1:])




if __name__ == "__main__":
    main()

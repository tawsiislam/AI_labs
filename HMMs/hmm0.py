# HMM0 Next Emission Distribution

# You will be given three matrices (in this order); transition matrix, emission matrix, and initial state probability distribution. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input.



# you should output the emission probability distribution on a single line in the same matrix format, including the dimensions.




outPut = [1, 3, 0.3, 0.6, 0.1]



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

    # split string into elements
    # elements = matrixStr.split()

    for rowIndx in range(nDim):
        row = []
        for colIndx in range(mDim):

            # add element to row: rowIndx: is the row number, mDim: is the number of columns, colIndx: is the column number
            row.append(matrixElements[rowIndx * mDim + colIndx])
        matrix.append(row)
    return matrix
    



def MatrixMultiplication(matA: list, matB: list):
    """
    Takes in two matrices and returns their product
    A: nDimA x mDimA
    B: nDimB x mDimB
    product: nDimA x mDimB

    requires mDimA == nDimB
    """
    
    #---- get dimensions of the matrices------#

    nDimA = len(matA)
    mDimA = len(matA[0])

    nDimB = len(matB)
    mDimB = len(matB[0])
    #-----------------------------------#

    # check if matrices can be multiplied
    if mDimA != nDimB:
        raise Exception("Matrices cannot be multiplied")


    # create empty matrix to store product
    product = [[0 for row in range(mDimB)] for col in range(nDimA)]
    # explanation: row is the number of rows in the product matrix, from 0 to nDimA
    #              col is the number of columns in the product matrix, from 0 to mDimB

    # multiply matrices
    for row in range(nDimA):
        for col in range(mDimB):
            for k in range(mDimA):
                product[row][col] += matA[row][k] * matB[k][col]
    return product





    
"""
4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5 
4 3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.2 0.6 0.2 
1 4 0.0 0.0 0.0 1.0 
"""


def main():
    # read the inputs:
    A = [float(x) for x in input().split()] # transition matrix
    B = [float(x) for x in input().split()] # emission matrix
    pi = [float(x) for x in input().split()] # initial state probability distribution

    # convert inputs to matrices
    

    A = ParseMatrix(A)
    B = ParseMatrix(B)
    pi = ParseMatrix(pi)


    # next state is A*pi
    xNew = MatrixMultiplication(pi, A) # return is list of lists
    # print("next state", xNew)

    # next emission is B*xNew
    outPut = MatrixMultiplication(xNew, B)
    # print("next emission", outPut)


    # insert sizeOfOutput at the beginning of outPut
    outPut[0].insert(0, len(outPut[0]))
    outPut[0].insert(0, len(outPut))
    
    print(str(outPut).replace("[", "").replace("]", "").replace(",", "").strip("'")) # remove brackets and commas from output

    





if __name__ == "__main__":
    main()

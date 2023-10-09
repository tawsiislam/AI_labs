from math import log
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
            row.append(float(matrixElements[rowIndx * mDim + colIndx]))
        matrix.append(row)
    return matrix
    
def outputMatrix(matrix):
    matrix_list = []
    matrix_list.append(len(matrix))
    matrix_list.append(len(matrix[0]))
    for row in matrix:
        for elem in row:
            matrix_list.append(round(elem,6))
    print(' '.join(map(str,matrix_list)))

def alphaPass(A: list, B: list, pi: list, O: list, scaling=True):   #Scaling suitable for Baum-Welch algorithm
    totStateIterable = range(len(A)) #Substitue repeating range(len(A)) to create a loop
    totObsIterable = range(len(O))

    alpha = [[0 for obs in totStateIterable] for state in totObsIterable]
    c = [0 for obs in totObsIterable]
    for state in totStateIterable:
        alpha[0][state] = pi[state]*B[state][O[0]]
        c[0] += alpha[0][state]
    c[0] = 1/(c[0]+sys.float_info.epsilon*0)
    if scaling == False:
        for state in totStateIterable:
            alpha[0][state] *= c[0]  #Normalising alpha[state][0]/c[0]

    for obs_t in totObsIterable[1:]:
        for state_i in totStateIterable:
            for state_j in totStateIterable:
                alpha[obs_t][state_i] = alpha[obs_t][state_i] + alpha[obs_t-1][state_j]*A[state_j][state_i]
            alpha[obs_t][state_i] *= B[state_i][O[obs_t]]
            c[obs_t] += alpha[obs_t][state_i]
        c[obs_t] = 1/(c[obs_t]+sys.float_info.epsilon*0)
        if scaling == True:
            for state_i in totStateIterable:
                alpha[obs_t][state_i] *= c[obs_t]

    return alpha, c 


def betaPass(A: list, B: list, O: list, c: list):
    totStateIterable = range(len(A)) #Substitue repeating range(len(A)) to create a loop
    totObsIterable = range(len(O))

    beta = [[0 for state in totStateIterable] for obs in totObsIterable]

    for state in totStateIterable:
        beta[-1][state] = c[-1] #Get last element. c here is not inversed

    for obs_t in reversed(totObsIterable[0:-1]):
        for state_i in totStateIterable:
            for state_j in totStateIterable:
                beta[obs_t][state_i] += A[state_i][state_j]*B[state_j][O[obs_t+1]]*beta[obs_t+1][state_j]
            beta[obs_t][state_i] *= c[obs_t]
    
    return beta

def gammaFunc(A: list, B: list, O: list, alpha: list, beta: list):
    totStateIterable = range(len(A))
    totObsIterable = range(len(O))
    gamma = [[0 for state in totStateIterable] for obs in totObsIterable]
    digamma = [[[0 for state_j in totStateIterable] for state_i in totStateIterable] for obs in totObsIterable]

    for obs_t in totObsIterable[:-1]:
        for state_i in totStateIterable:
            for state_j in totStateIterable:
                digamma[obs_t][state_i][state_j] = alpha[obs_t][state_i]*A[state_i][state_j]*B[state_j][O[obs_t+1]]*beta[obs_t+1][state_j]
                gamma[obs_t][state_i] += digamma[obs_t][state_i][state_j]

    for state_i in totStateIterable:
        gamma[-1][state_i] = alpha[-1][state_i]

    return gamma, digamma

def reestimate(A: list, B: list, pi: list, O: list, gamma: list, digamma: list):
    totStateIterable = range(len(A))
    totObsIterable = range(len(O))
    totMStateIterable = range(len(B[0]))

    new_pi = [0 for state in totStateIterable]
    new_A = [[0 for state in totStateIterable] for state in totStateIterable]
    new_B = [[0 for obs in range(len(B[0]))] for state in range(len(B))]

    for state_i in totStateIterable:
        new_pi[state_i] = gamma[0][state_i] #Some elements in gamma causes there to be only one non-zero element
    
    for state_i in totStateIterable:    # Re-estimate A
        denom = 0
        for obs_t in totObsIterable[:-1]:
            denom += gamma[obs_t][state_i]
        for state_j in totStateIterable:
            numer = 0
            for obs_t in totObsIterable[:-1]:
                numer += digamma[obs_t][state_i][state_j]
            new_A[state_i][state_j] = numer/(denom+sys.float_info.epsilon*0)
    
    for state_i in totStateIterable:    # Re-estimate B
        denom = 0
        for obs_t in totObsIterable:
            denom += gamma[obs_t][state_i]
        for state_j in totMStateIterable:
            numer = 0
            for obs_t in totObsIterable:
                if (O[obs_t]==state_j):
                    numer += gamma[obs_t][state_i]
            new_B[state_i][state_j] = numer/(denom+sys.float_info.epsilon*0)
    return new_A, new_B, new_pi

def logProbFunc(c: list, lenO: int):
    totObsIterable = range(lenO)
    logProb = 0
    for obs_t in totObsIterable:
        logProb += log(c[obs_t])  #Uses math.log
    
    return -logProb

def BaumWelch_Algo(A: list, B: list, pi: list, O: list, max_iter: int=10):
    prevLogProb = float('-inf')
    iter = 0
    while(True):
        alpha,c = alphaPass(A, B, pi, O)
        beta = betaPass(A, B, O, c)
        gamma, digamma = gammaFunc(A, B, O, alpha, beta)
        A, B, pi = reestimate(A, B, pi, O, gamma, digamma)
        newLogProb = logProbFunc(c,len(O))
        if  newLogProb > prevLogProb or abs(newLogProb - prevLogProb) < 0.0001*prevLogProb: #TODO: this was added for Cpart               or abs(newLogProb - prevLogProb) < 0.0001*prevLogProb
            prevLogProb = newLogProb
        else:
            break

        if(iter%200 ==0): print("iter ", iter)
    
    print("iter ", iter)

    return A, B, pi

def main():
    # read the inputs:
    A = [float(x) for x in input().split()] # transition matrix
    B = [float(x) for x in input().split()] # emission matrix
    pi = [float(x) for x in input().split()] # initial state probability distribution
    emissionSequence = [int(x) for x in input().split()] # observation sequence
    
    # convert inputs to matrices
    A = ParseMatrix(A)
    B = ParseMatrix(B)
    pi = ParseMatrix(pi)
    emissionSequence = emissionSequence[1:]


    #---------------------------------------------------------------------------
    A, B, pi = BaumWelch_Algo(A, B, pi[0], emissionSequence)
    outputMatrix(A)
    outputMatrix(B)
    









# C part 

# Q7
def testQ7():
    Aexpected = "3 3 0.7 0.05 0.25 0.1 0.8 0.1 0.2 0.3 0.5"
    Bexpected = "3 4 0.7 0.2 0.1 0.0 0.1 0.4 0.3 0.2 0.0 0.1 0.2 0.7"
    piexpected = "1 3 1 0 0"

    Aexpected = [float(x) for x in Aexpected.split()] # transition matrix
    Bexpected = [float(x) for x in Bexpected.split()] # emission matrix
    piexpected = [float(x) for x in piexpected.split()] # initial state probability distribution


    A_init = "3 3 0.54 0.26 0.20 0.19 0.53 0.28 0.22 0.18 0.6"
    B_init = "3 4 0.5 0.2 0.11 0.19 0.22 0.28 0.23 0.27 0.19 0.21 0.15 0.45"
    pi_init = "1 3 0.3 0.2 0.5"

    A_init = [float(x) for x in A_init.split()] # transition matrix
    B_init = [float(x) for x in B_init.split()] # emission matrix
    pi_init = [float(x) for x in pi_init.split()] # initial state probability distribution

    
    # convert to matrices
    Aexpected = ParseMatrix(Aexpected)
    Bexpected = ParseMatrix(Bexpected)
    piexpected = ParseMatrix(piexpected)

    A = ParseMatrix(A_init)
    B = ParseMatrix(B_init)
    pi = ParseMatrix(pi_init)

    # load data structute hmm_c_N1000.in
    emissionSequence = []
    with open("hmm_c_N10000.in") as f: # n = 1000 converges in 3930 iterations
        for line in f:
            emissionSequence.append((line))
    newEmmisionSequence = []
    # split all lines into a list of integers
    for line in emissionSequence:
        newEmmisionSequence.append([int(x) for x in line.split()])
    # print("emissionSeq ", newEmmisionSequence[0][1:])

    # calling on baum welch algorithm
    A, B, pi = BaumWelch_Algo(A, B, pi[0], newEmmisionSequence[0][1:], 10000)
    outputMatrix(A)
    outputMatrix(B)












if __name__ == "__main__":
    # main()
    testQ7()











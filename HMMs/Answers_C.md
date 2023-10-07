# Q7
With 1000 observations it ended after 3931 iterations with A and B \
3 3 0.696474 0.013356 0.29017 0.101464 0.812012 0.086523 0.192118 0.301278 0.506604 \
3 4 0.688799 0.225157 0.075369 0.010674 0.067868 0.412067 0.281392 0.238673 0.0 0.0 0.353301 0.646699

For 10000 observations it ended after 15671 iterations with A and B\
3 3 0.694281 0.044899 0.26082 0.117668 0.746077 0.136255 0.154186 0.256694 0.58912 \
3 4 0.71 0.186409 0.103592 0.0 0.098812 0.421125 0.312174 0.167889 0.032113 0.171325 0.186628 0.609933

We saw that both cases converged after many iterations. They both didn't converge to what the actual model was, but the results from 10000 observations were closer to the actual model. This can be explained as this is a statistical model derived from EM, where more data will give better results. However, in the end the estimation will be better the closer our initialisation are to the actual model as there could be several different local minima that we could converge to.

You could define the convergence a point where the probability to get our observations using the estimated model doesn't improve. This is a point where you reach local minimum from the initilisation you made.

# Q8
Starting with \
3 3 0.65 0.06 0.29 0.13 0.75 0.12 0.15 0.26 0.59 \
3 4 0.75 0.07 0.179 0.01 0.07 0.37 0.32 0.24 0.01 0.12 0.22 0.659 \
1 3 0.99 0.005 0.005\
we get 
3 3 0.694281 0.044899 0.260820 0.117668 0.746077 0.136255 0.154186 0.256694 0.589120 \
3 4 0.710000 0.186409 0.103592 0.0 0.098812 0.421125 0.312174 0.167889 0.032113 0.171325 0.186628 0.609933

When starting close to the actual we got the same result as before when running with 10000 observations. Now there could be some slight differences if we added more decimal places without rounding. However, for most models, having such an accurate values are insignificant and you would possibly overfit. We cannot even say if the model is better or not when the answer only has two decimal places. We saw the same result when using 1000 observations.

The problem with estimating the distance is what choice of methods is suitable for our case. You could either sum up the absolute values of the difference between the elements, find the norm using all elements, or compare the element with largest difference. First two are more suitable to not allow an outling element decide how close the matrices are. Sometime one element might not affect the overall model if other elements are good enough. 

# Q9
We used following matrices where we reduced to 2 states.
2 2 0.45 0.55 0.52 0.48
2 4 0.22 0.27 0.19 0.32 0.26 0.18 0.23 0.33
1 2 1 0
The result we got was 
2 2 0.8517440 0.148256 0.340222 0.659778
2 4 0.061541 0.314214 0.259754 0.364490 0.729334 0.168192 0.090863 0.011610.

Issue with having less states would be that you lose information on how the data was generated. Having too much would be overfitting and not handle noisy observations well.
If we know that 3x3 A matrix and 3x4 matrix generated the data then these parameters are the best to start with when initialising our matrices.
You cannot find the correct parameters from the start without doing some trial-and-error and evaluate your model in order to find the optimal settings. The more data you have available to better your evaluation will be able to tell you how good your settings are. 

# Q10
## Uniform distribution
Final model
3 3 0.333334 0.333333 0.333334 0.333335 0.333334 0.333332 0.333334 0.333333 0.333334\
3 4 0.2642 0.2699 0.2085 0.2574 0.2642 0.2699 0.2085 0.2574 0.2642 0.2699 0.2085 0.2574\
Starting with a uniform distribution gives the same matrix for A but with some difference for B. This because this result of a model is a local minimum where all the state are transitioning with equal probailities. Distinguishing  the different states from each other using the data becomes difficult and there are no improvement on the model. 

## Diagonal matrix
3 3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\
3 4 0.2642 0.2699 0.2085 0.2574 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\
Because we are working with a lot of zeros, the result we saw is that you have a lot of NaN values. This is due to there occuring division by zero, which is undefined. This undefined value is then multiplied with all the other elements and occur in the gamma that causes problem when trying to reestimate the matrices.

## Close to solution
Starting with 
3 3 0.699 0.041 0.26 0.099 0.791 0.11 0.199 0.291 0.51
3 4 0.699 0.201 0.099 0.001 0.099 0.391 0.305 0.205 0.001 0.099 0.199 0.701
1 3 0.999 0.0005 0.0005

When having close to the solution we are getting same results when working with 10000 observations. You could say the error is caused by the noise in the observation that causes the estimation of the model to not be perfect or there are inaccuracies during the calculations. The error becomes smaller the more data we have as Baum-Welch is statistical model. If the behaviour of the noise is known then you could add that to be part of the model for better estimations. However, in practice, it's difficult to model the noise.
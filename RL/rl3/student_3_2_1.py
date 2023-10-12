#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
#TODO Run and play again with hyperparameters in order to understand them
rewards = [10, -10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10] #TODO: ask if all rewards are -10, why is the second last one 10?   

# Q learning learning rate
alpha = 0.15

# Q learning discount rate
gamma = 0.15

# Epsilon initial
epsilon_initial = 1

# Epsilon final
epsilon_final = 1

# Annealing timesteps
annealing_timesteps = 1

# threshold
threshold = 1e-6

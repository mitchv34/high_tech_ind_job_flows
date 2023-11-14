#==========================================================================================
Title: Rational Inattention Priors
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-11-11
Description: This file contains the code to compute the rational inattention unconditional 
    choice probabilities. Based on Matejka and McKay (2015).
==========================================================================================#

# Patameters
R = 0.5
N = 2                   # Number of choices
values = [R, [0, 1]]    # Values for each choice
λ = 0.5                 # Cost of attention

n_states = [length(v) for v in values] # Number of states for each choice
# Prior beliefs should be a vecter of length N with each element being a vector of length n_states[i]
# For simplicity we will assume that the prior is uniform over the states
prior = [ones(n_states[i])/n_states[i] for i in 1:N]

# Compute coeficients fot the optional strategy for given realized vector of states V
compute_coefs(V) = [exp(v / λ) for v in V]


# Compute coeficients for all possible vectors of states
coefs = [compute_coefs(v) for v in Iterators.product(values...)]

all_states_prob = prod.(collect(Iterators.product(prior...))[:])




x = [['a', 'b'], ['c', 'd'],['e','f']]  # Replace with your vector of vectors

# Compute the Cartesian product of the vectors in x
# products = [p... for p in Iterators.product(x...)

prod.(collect(Iterators.product(x...))[:])
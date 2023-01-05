# hidden markov chain
from hmm_from_scratch.model import ProbabilityVector, ProbabilityMatrix, HiddenMarkovChain, HiddenMarkovChain_FP

a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})

A = ProbabilityMatrix({'1H': a1, '2C': a2})
B = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

print(A)
print(A.df)
print(pi.df)

print(B)
print(B.df)

hmc = HiddenMarkovChain(A, B, pi)
observations = ['1S', '2M', '3L', '2M', '1S']
print("Score for {} is {:f}.".format(observations, hmc.score(observations)))

# # results with pen and paper
# observations = ['1S', '3L']  # output: 0.095600
# p_1h1h = 0.6 * 0.1 * 0.7 * 0.5
# p_1h2c = 0.6 * 0.1 * 0.3 * 0.1
# p_2c1h = 0.4 * 0.7 * 0.4 * 0.5
# p_2c2c = 0.4 * 0.7 * 0.6 * 0.1
# hand_score = p_1h1h + p_1h2c + p_2c1h + p_2c2c
# print(fr"Score obtained using pen and paper: {hand_score}")
# print("Score for {} is {:f}.".format(observations, hmc.score(observations)))

hmc_fp = HiddenMarkovChain_FP(A, B, pi)
observations = ['1S', '2M', '3L', '2M', '1S']
print("Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))


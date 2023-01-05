# latent states
from hmm_from_scratch.model import ProbabilityVector, ProbabilityMatrix, HiddenMarkovChain_Simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

hmc_s = HiddenMarkovChain_Simulation(A, B, pi)
stats = {}
for length in np.logspace(1, 5, 40).astype(int):
    observation_hist, states_hist = hmc_s.run(length)
    stats[length] = pd.DataFrame({'observations': observation_hist,
                                  'states': states_hist}).applymap(lambda x: int(x[0]))

S = np.array(list(map(lambda x: x['states'].value_counts().to_numpy() / len(x), stats.values())))

plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
plt.xlabel('Chain length T')
plt.ylabel('Probability')
plt.title('Converging probabilities')
plt.legend(['1C', '2C'])
plt.show()

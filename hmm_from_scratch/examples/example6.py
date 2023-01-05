# backward pass
from hmm_from_scratch.model import ProbabilityVector, ProbabilityMatrix, HiddenMarkovChain_Backward
import numpy as np
import pandas as pd
from itertools import product

np.random.seed(42)

a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})
b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})
A = ProbabilityMatrix({'1H': a1, '2C': a2})
B = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

hmc = HiddenMarkovChain_Backward(A, B, pi)

observed_sequence, latent_sequence = hmc.run(5)
uncovered_sequence = hmc.backward(observed_sequence)

data = [observed_sequence, latent_sequence, uncovered_sequence]
df = pd.DataFrame(data, index=['observation', 'latent (actual) states', 'most likely states'])
print(df)

# likelihood of different latent sequences
all_possible_states = {'1H', '2C'}
chain_length = 6  # any int > 0
all_states_chains = list(product(*(all_possible_states,) * chain_length))

df = pd.DataFrame(all_states_chains)
dfp = pd.DataFrame()

for i in range(chain_length):
    dfp['p' + str(i)] = df.apply(lambda x: hmc.B.df.loc[x[i], observed_sequence[i]], axis=1)  # all possible B sequences

scores = dfp.sum(axis=1).sort_values(ascending=False)
df = df.iloc[scores.index]
df['score'] = scores
df.head(10).reset_index()
print(df)
print(df.shape)

# find the (actual) latent sequence that caused the observations is on the 34th position (starting from index 0)
dfc = df.copy().reset_index()
for i in range(chain_length):
    dfc = dfc[dfc[i] == latent_sequence[i]]
print(dfc)
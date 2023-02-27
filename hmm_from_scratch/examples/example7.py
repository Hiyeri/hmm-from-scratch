# training model
import numpy as np
from hmm_from_scratch.model import HiddenMarkovModel, HiddenMarkovLayer
import matplotlib.pyplot as plt

np.random.seed(42)

observations = ['3L', '2M', '1S', '3L', '3L', '3L']

states = ['1H', '2C']
observables = ['1S', '2M', '3L']

hml = HiddenMarkovLayer.initialize(states, observables)
hmm = HiddenMarkovModel(hml)

hmm.train(observations, 25)
score_history = hmm.score_history
plt.plot(score_history)
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training HMM on a 6-element observation sequence')
plt.show()
from itertools import product
from functools import reduce
from .probabilities import ProbabilityVector, ProbabilityMatrix


class HiddenMarkovChain:
    def __init__(self, A, B, pi):
        self.A = A  # transmission matrix A
        self.B = B  # emission matrix B
        self.pi = pi  # initial state distribution
        self.states = pi.states
        self.observables = B.observables

    def __repr__(self):
        return "HML states; {} -> observables: {}.".format(len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        A = ProbabilityMatrix.initialize(states, states)
        B = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(A, B, pi)

    def __create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self.__create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.A.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.B.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.A.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


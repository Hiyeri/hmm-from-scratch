from .hmc_backward import HiddenMarkovChain_Backward
import numpy as np


class HiddenMarkovLayer(HiddenMarkovChain_Backward):
    def _xi(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        xi = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)  # p(O|lambda)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.A.values)
            P2 = self.B[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            xi[t, :, :] = P1 * P2 / score
        return xi

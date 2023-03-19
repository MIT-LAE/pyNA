
import numpy as np


def compute_frequency_subbands(f: np.ndarray, n_frequency_subbands: int) -> np.ndarray:
        """

        :param f: 
        :type f: np.ndarray
        :param n_frequency_subbands:
        :type n_frequency_subbands: int

        :return: None
        """

        # Calculate subband frequencies [Hz]
        # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 6-7
        # Source: Berton 2021 Simultaneous use of Ground Reflection and Lateral Attenuation Noise Models Appendix A Eq. 1
        f_sb = np.zeros(n_frequency_subbands * np.size(f))
        m = (n_frequency_subbands - 1) / 2.
        w = 2. ** (1 / (3. * n_frequency_subbands))

        for i, f_i in enumerate(f):
            for j in np.arange(n_frequency_subbands):
                f_sb[i * n_frequency_subbands + j] = w ** (j - m) * f_i

        return f_sb
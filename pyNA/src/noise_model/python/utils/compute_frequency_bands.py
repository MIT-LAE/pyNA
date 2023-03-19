import numpy as np


def compute_frequency_bands(n_frequency_bands: int, l_i=16) -> np.ndarray:
    """
    Compute the 1/3rd order frequency bands and with sub-bands.
        * f:    1/3rd order frequency bands
        * f_sb: frequency sub-bands

    :param n_frequency_bands: 
    :type n_frequency_bands: int
    :param l_i: Starting no. of the frequency band [-]
    :type n_i: int

    :return: None
    """

    # Generate 1/3rd octave frequency bands [Hz]
    f = 10 ** (0.1 * np.linspace(1+l_i, 40, n_frequency_bands))

    return f
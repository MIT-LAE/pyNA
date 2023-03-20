import numpy as np


def get_frequency_bands(n_frequency_bands: int, l_i=16) -> np.ndarray:
    """
    Get array of 1/3rd octave frequency bands

    Parameters
    ----------        
    n_frequency_bands : int
        Number of 1/3rd octave frequency bands
    l_i: int
        Starting no. of the frequency band [-]
    
    Output
    ------
    f : jnp.ndarray
        Array of 1/3rd octave frequency bands

    """

    # Generate 1/3rd octave frequency bands [Hz]
    f = 10 ** (0.1 * np.linspace(1+l_i, 40, n_frequency_bands))

    return f
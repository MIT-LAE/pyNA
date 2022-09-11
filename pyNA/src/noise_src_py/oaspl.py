import numpy as np

def oaspl(levels, spl: np.ndarray) -> np.ndarray:
    """
    Compute overall sound pressure level.

    :param levels: pyNA component computing noise levels
    :type levels: Levels
    :param spl: sound pressure level [dB]
    :type spl: np.ndarray

    :return: oaspl
    :rtype: np.ndarray

    """

    # Load options
    n_t = levels.options['n_t']

    # Compute OASPL by summing SPL logarithmically
    oaspl = np.zeros(n_t)
    for i in np.arange(n_t):
        oaspl[i] = 10 * np.log10(np.sum(10 ** (spl[i, :] / 10.)))

    return oaspl

import numpy as np

def spl(levels, msap_prop: np.ndarray, rho_0: np.ndarray, c_0: np.ndarray) -> np.ndarray:
    """
    Compute sound pressure level.

    :param levels: pyNA component computing noise levels
    :type levels: Levels
    :param msap_prop: mean-square acoustic pressure [-]
    :type msap_prop: np.ndarray [n_t, settings.N_f]
    :param rho_0: ambient density [kg/m3]
    :type rho_0: np.ndarray [n_t]
    :param c_0: ambient speed of sound [m/s]
    :type c_0: np.ndarray [n_t]

    :return: spl [dB]
    :rtype: np.ndarray [n_t, settings.N_f]

    """
    # Load options
    n_t = levels.options['n_t']

    # Compute SPL
    spl = 10*np.log10(msap_prop) + 20.*np.log10(np.reshape(rho_0, (n_t, 1)) * np.reshape(c_0, (n_t, 1)) ** 2.)

    # Clip values to avoid (spl < 0)
    spl = spl.clip(min=1e-99)

    return spl


import numpy as np


def calculate_spl(msap_prop, rho_0, c_0):
    """
    Compute sound pressure level.

    :param msap_prop: mean-square acoustic pressure [-]
    :type msap_prop
    :param rho_0: ambient density [kg/m3]
    :type rho_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0

    :return: spl [dB]
    :rtype

    """

    # Compute SPL
    spl = 10.*np.log10(msap_prop) + 20.*np.log10(rho_0 * c_0 ** 2)

    # Clip values to avoid (spl < 0)
    spl = spl.clip(min=1e-99)

    return spl


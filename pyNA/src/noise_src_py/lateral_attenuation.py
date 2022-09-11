import numpy as np
from typing import Dict, Any


def lateral_attenuation(settings: Dict[str, Any], beta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
    """
    Compute lateral attenuation coefficients.

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param beta: elevation angle [deg]
    :type beta: np.ndarray
    :param x_obs: observer location [m, m, m]
    :type x_obs: np.ndarray

    :return: Lambda
    :type: np.ndarray
    """

    # Depression angle: phi_d = beta (elevation angle) + epsilon (aircraft bank angle = 0)
    phi_d = beta

    # Lateral side distance
    l = x_obs[1]

    # Engine installation term [dB]
    if settings['lateral_attenuation_engine_mounting'] == "underwing":
        E_eng = 10 * np.log10((0.0039 * np.cos(phi_d) ** 2 + np.sin(phi_d) ** 2) ** 0.062 / (0.8786 * np.sin(2 * phi_d) ** 2 + np.cos(2 * phi_d) ** 2))
    elif settings['lateral_attenuation_engine_mounting'] == "fuselage":
        E_eng = 10 * np.log10((0.1225 * np.cos(phi_d) ** 2 + np.sin(phi_d) ** 2) ** 0.329)
    elif settings['lateral_attenuation_engine_mounting'] == "propeller":
        E_eng = 0.
    elif settings['lateral_attenuation_engine_mounting'] == "none":
        E_eng = 0.
    else:
        raise ValueError('Invalid engine_mounting specified. Specify: underwing/fuselage/propeller/none.')

    # Attenuation caused by ground and refracting-scattering effects [dB]
    if beta <= 50.:
        A_grs = (1.137 - 0.0229 * beta + 9.72 * np.exp(-0.142 * beta))
    else:
        A_grs = 0.

    # Over-ground attenuation [dB]
    if 0. <= l <= 914:
        g = 11.83 * (1 - np.exp(-0.00274 * l))
    elif l > 914:
        g = 10.86  # 11.83*(1-np.exp(-0.00274*914))
    else:
        raise ValueError('Lateral sideline distance negative.')

    # Overall lateral attenuation
    Lambda = 10 ** ((E_eng - g * A_grs / 10.86) / 10.)

    return Lambda

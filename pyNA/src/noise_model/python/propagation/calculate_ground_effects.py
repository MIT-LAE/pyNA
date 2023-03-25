import numpy as np
from scipy import special


def calculate_ground_effects(msap_sb: np.ndarray, rho_0: float, c_bar: float, r: float, beta: float, f_sb: np.ndarray, x_mic: np.ndarray, settings: dict) -> np.ndarray:

    """
    
    Arguments
    ---------
    msap_sb : np.ndarray
        _
    rho_0 : float
        _
    c_bar : float
        -
    r : float
        _
    beta : float
        _
    f_sb : np.ndarray
        _
    x_mic : np.ndarray
        _
    settings : dict
        _

    Outputs
    -------
    msap_ge : np.ndarray
        _

    """

    # Compute difference in direct and reflected distance between source and observer
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 16
    r_r = np.sqrt(r ** 2 + 4 * x_mic[2] ** 2 + 4 * r * x_mic[2] * np.sin(beta * np.pi / 180.))
    dr = r_r - r

    # Compute wave number
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * np.pi * f_sb / c_bar

    # Compute dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * np.pi * rho_0 * f_sb / settings['ground_resistance']

    # Compute the cosine of the incidence angle
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 18
    cos_theta = (r * np.sin(beta * np.pi / 180.) + 2 * x_mic[2]) / r_r

    # Complex specific ground admittance nu
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 13 / adapted through Berton lateral attenuation paper
    nu = (1 + (6.86 * eta) ** (-0.75) + (4.36 * eta) ** (-0.73) * 1j) ** (-1)

    # Compute Gamma
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 5
    Gamma = (cos_theta - nu) / (cos_theta + nu)

    # Compute tau
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 9
    tau = np.sqrt(k * r_r / 2j) * (cos_theta + nu)

    # Compute complex spherical wave reflection coefficient
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 12
    U = np.real(tau ** 0)
    U[np.where(-np.real(tau) == 0)] = 0.5 * np.ones(settings['n_frequency_subbands'] * settings['n_frequency_bands'], dtype=np.float64)[np.where(-np.real(tau) == 0)]
    U[np.where(-np.real(tau) < 0)] = np.zeros(settings['n_frequency_subbands'] * settings['n_frequency_bands'], dtype=np.float64)[np.where(-np.real(tau) < 0)]

    # Compute F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
    F = -2 * np.sqrt(np.pi) * U * tau * np.exp(tau ** 2) + 1. / (2. * tau ** 2) - 3. / (2 * tau ** 2) ** 2

    # Compute complex spherical wave function F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 10 (using Faddeeva function)
    F[np.where(np.absolute(tau)<10)] = (1 - np.sqrt(np.pi) * tau * (special.wofz(tau * 1j)))[np.where(np.absolute(tau)<10)]

    # Compute Z_cswfc
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
    Z_cswfc = Gamma + (1 - Gamma) * F
    R = np.absolute(Z_cswfc)
    alpha = np.angle(Z_cswfc)

    # Compute the constant K and constant epsilon
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
    K = 2 ** (1. / (6. * settings['n_frequency_subbands']))
    eps = K - 1

    # Compute G
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
    if dr > 0:
        msap_ge = (1 + R ** 2 + 2 * R * np.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * np.cos(alpha + k * dr) * np.sin(eps * k * dr) / (eps * k * dr)) * msap_sb
    else:
        msap_ge = (1 + R ** 2 + 2 * R * np.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * np.cos(alpha + k * dr)) * msap_sb

    return msap_ge

import numpy as np
from typing import Dict, Any
from scipy import special
from pyNA.src.data import Data
import pdb


def ground_effects(settings: Dict[str, Any], data: Data, r: np.float64, beta: np.float64, x_obs: np.ndarray, c_bar: np.float64, rho_0: np.float64) -> np.ndarray:
    """
    Compute the ground reflection coefficients.

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param data: pyna noise data
    :type data: Data
    :param r: distance source to observer [m]
    :type r: np.float64
    :param beta: elevation angle [deg]
    :type beta: np.float64
    :param x_obs: observer location [m, m, m]
    :type x_obs: np.ndarray
    :param c_bar: average ambient speed of sound between observer and source [m/s]
    :type c_bar: np.float64
    :param rho_0: ambient density [kg/m3]
    :type rho_0: np.float64

    :return: G
    :rtype: np.ndarray

    """

    # Compute difference in direct and reflected distance between source and observer
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 16
    r_r = np.sqrt(r ** 2 + 4 * x_obs[2] ** 2 + 4 * r * x_obs[2] * np.sin(beta * np.pi / 180.))
    dr = r_r - r

    # Compute wave number
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * np.pi * data.f_sb / c_bar

    # Compute dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * np.pi * rho_0 * data.f_sb / settings['ground_resistance']

    # Compute the cosine of the incidence angle
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 18
    cos_theta = (r * np.sin(beta * np.pi / 180.) + 2 * x_obs[2]) / r_r

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
        G = 1 + R ** 2 + 2 * R * np.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * np.cos(alpha + k * dr) * np.sin(eps * k * dr) / (eps * k * dr)
    else:
        G = 1 + R ** 2 + 2 * R * np.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * np.cos(alpha + k * dr)

    return G

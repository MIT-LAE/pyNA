import jax
import jax.numpy as jnp


def compute_ground_effects(msap_sb, rho_0, c_bar, r, beta, f_sb, x_obs, settings):

    # Compute difference in direct and reflected distance between source and observer
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 16
    r_r = jnp.sqrt(r ** 2 + 4 * x_obs[2] ** 2 + 4 * r * x_obs[2] * jnp.sin(beta * jnp.pi / 180.))
    dr = r_r - r

    # Compute wave number
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * jnp.pi * f_sb / c_bar

    # Compute dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * jnp.pi * rho_0 * f_sb / settings['ground_resistance']

    # Compute the cosine of the incidence angle
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 18
    cos_theta = (r * jnp.sin(beta * jnp.pi / 180.) + 2 * x_obs[2]) / r_r

    # Complex specific ground admittance nu
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 13 / adapted through Berton lateral attenuation paper
    nu = (1 + (6.86 * eta) ** (-0.75) + (4.36 * eta) ** (-0.73) * 1j) ** (-1)

    # Compute Gamma
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 5
    Gamma = (cos_theta - nu) / (cos_theta + nu)

    # Compute tau
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 9
    tau = jnp.sqrt(k * r_r / 2j) * (cos_theta + nu)

    # Compute complex spherical wave reflection coefficient
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 12
    # U = jnp.real(tau ** 0)
    # U = U.at[jnp.where(-jnp.real(tau) == 0)] = 0.5 * jnp.ones(settings['n_frequency_subbands'] * settings['n_frequency_bands'], dtype=jnp.float64)[jnp.where(-jnp.real(tau) == 0)]
    # U = U.at[jnp.where(-jnp.real(tau) < 0)] = jnp.zeros(settings['n_frequency_subbands'] * settings['n_frequency_bands'], dtype=jnp.float64)[jnp.where(-jnp.real(tau) < 0)]

    # Compute F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
    # F = -2 * jnp.sqrt(jnp.pi) * U * tau * jnp.exp(tau ** 2) + 1. / (2. * tau ** 2) - 3. / (2 * tau ** 2) ** 2

    # Compute complex spherical wave function F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 10 (using Faddeeva function)
    # wofz = jnp.exp(tau**2) * jax.scipy.special.erfc(tau)
    # F[jnp.where(jnp.absolute(tau)<10)] = (1 - jnp.sqrt(jnp.pi) * tau * wofz)[jnp.where(jnp.absolute(tau)<10)]

    # TODO: implement Faddeeva function 
    wofz = 1
    F = 1. - jnp.sqrt(jnp.pi) * tau * wofz

    # Compute Z_cswfc
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
    Z_cswfc = Gamma + (1 - Gamma) * F
    R = jnp.absolute(Z_cswfc)
    alpha = jnp.angle(Z_cswfc)

    # Compute the constant K and constant epsilon
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
    K = 2 ** (1. / (6. * settings['n_frequency_subbands']))
    eps = K - 1

    # Compute G
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
    if dr > 0:
        G = (1 + R ** 2 + 2 * R * jnp.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * jnp.cos(alpha + k * dr) * jnp.sin(eps * k * dr) / (eps * k * dr)) * msap_sb
    else:
        G = (1 + R ** 2 + 2 * R * jnp.exp(-(settings['incoherence_constant'] * k * dr) ** 2) * jnp.cos(alpha + k * dr)) * msap_sb

    return G

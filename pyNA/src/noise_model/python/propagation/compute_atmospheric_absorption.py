import jax.numpy as jnp


def compute_atmospheric_absorption(msap_sb, r, z, f_sb, settings, tables):

    """
    
    :param r:
    :type r:
    :param z:
    :type z:
    :param f_sb:
    :type f_sb:
    :param settings:
    :type settings:
    :param tables:
    :type tables:

    """

    # ---------- Apply atmospheric absorption on sub-bands ----------
    # Compute average absorption factor between observer and source
    alpha_f = tables.propagation.abs_f(f_sb, z)

    # Compute absorption (convert dB to Np: 1dB is 0.115Np)
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
    msap_abs = msap_sb * jnp.exp(-2 * 0.115 * alpha_f * (r - settings['r_0']))

    return msap_abs

import numpy as np
from pyNA.src.noise_model.tables import Tables


def calculate_atmospheric_absorption(msap_sb: np.ndarray, r: float, z: float, f_sb: np.ndarray, settings: dict, tables: Tables) -> np.ndarray:

    """
    
    Arguments
    ---------
    msap_sb : np.ndarray
        _
    r : float
        _
    z : float
        _
    f_sb : np.ndarray
        _
    settings : dict
        _
    tables : Tables
        _

    Outputs
    -------
    msap_abs : np.ndarray
        _

    """

    # Apply atmospheric absorption on sub-bands
    # Compute average absorption factor between observer and source
    alpha_f = tables.propagation.abs_f(f_sb, z)

    # Compute absorption (convert dB to Np: 1dB is 0.115Np)
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
    msap_abs = msap_sb * np.exp(-2 * 0.115 * alpha_f * (r - settings['r_0']))

    return msap_abs

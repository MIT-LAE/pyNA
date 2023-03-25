import numpy as np
import pdb


def calculate_direct_propagation(msap_source: np.ndarray, r: float, I_0: float, settings: dict) -> np.ndarray:

    """
    
    Arguments
    ---------
    msap_source : np.ndarray
        _
    r : float
        _
    I_0 : float
        _
    settings : dict
        _

    Outputs
    -------
    msap_dp : np.ndarray
        _
    
    """

    I_0_obs = 409.74  # Sea level characteristic impedance [kg/m**2/s]
    msap_dp = msap_source * (settings['r_0']/r)**2 * (I_0_obs / I_0)

    return msap_dp
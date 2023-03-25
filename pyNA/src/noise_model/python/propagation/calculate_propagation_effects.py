import numpy as np
from pyNA.src.noise_model.python.propagation.calculate_direct_propagation import calculate_direct_propagation
from pyNA.src.noise_model.python.propagation.calculate_atmospheric_absorption import calculate_atmospheric_absorption
from pyNA.src.noise_model.python.propagation.calculate_ground_effects import calculate_ground_effects
from pyNA.src.noise_model.python.propagation.calculate_lateral_attenuation import calculate_lateral_attenuation
from pyNA.src.noise_model.python.utils.get_msap_subbands import get_msap_subbands
from pyNA.src.aircraft import Aircraft
from pyNA.src.noise_model.tables import Tables



def calculate_propagation_effects(msap_source: np.ndarray, z: float, r: float, c_bar: float, rho_0: float, I_0: float, beta: float, 
                                  x_mic: np.ndarray, f_sb: np.ndarray, settings: dict, tables: Tables) -> np.ndarray:

    """
    
    Arguments
    ---------
    msap_source : np.ndarray
        _
    x : float
        _
    y : float
        _
    z : float
        _
    r: float
        _
    c_bar : float
        _
    rho_0 : float
        _
    I_0 : float
        _
    beta: float 
        _
    x_mic : np.ndarray
        _
    f_sb : np.ndarray
        _
    settings : dict
        _
    aircraft : Aircraft
        _
    tables : Tables
        _

    Outputs
    -------
    msap_prop : np.ndarray
        _

    """

    # Apply spherical spreading and characteristic impedance effects to the MSAP
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
    if settings['direct_propagation']:
        msap_dp = calculate_direct_propagation(msap_source, r, I_0, settings)
    else:
        msap_dp = msap_source

    # Generate msap sub-bands
    msap_prop = np.zeros(settings['n_frequency_bands'])

    if settings['absorption'] or settings['ground_effects']:
        msap_sb = get_msap_subbands(msap_in=msap_dp, settings=settings)

        if settings['absorption']:
            msap_sb = calculate_atmospheric_absorption(msap_sb, r, z, f_sb, settings, tables)
        
        if settings['ground_effects']:
            msap_sb = calculate_ground_effects(msap_sb, rho_0, c_bar, r, beta, f_sb, x_mic, settings)
    
        if settings['lateral_attenuation']:
            pass
            # calculate_lateral_attenuation(beta, x_mic, settings)
        else:
            pass

        for j in np.arange(settings['n_frequency_bands']):
            msap_prop[j] = np.sum(msap_sb[j*settings['n_frequency_subbands']:(j+1)*settings['n_frequency_subbands']])
    
    else:
        msap_prop = msap_dp

    return msap_prop
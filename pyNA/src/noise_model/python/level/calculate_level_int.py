import numpy as np


def calculate_level_int(t_o: np.ndarray, level: np.ndarray, settings: dict):

    """
    
    Parameters
    ----------
    t_o : np.ndarray
        _
    level : np.ndarray
        _
    settings : dict
        pyna_settings

    """
    
    # Interpolate time, level
    n_ip = np.int64(np.ceil((level[-1]-level[0])/settings["epnl_dt"]))

    t_ip = np.zeros(n_ip, )
    for i in np.arange(n_ip, step=1):
        t_ip = t_ip.at[i].set(level[1] + (i-1)*settings["epnl_dt"])

    # Interpolate the data
    level_ip = np.interp(t_ip, t_o, level)
   
    # Compute max. level
    level_max = np.max(level_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10**(level_ip / 10.)

    # Compute integration bounds            
    D = 10 * np.log10(np.sum(f_int)) - level_max - 10 * np.log10(10. / settings["epnl_dt"])

    # Compute ilevel
    ilevel = level_max + D

    return ilevel


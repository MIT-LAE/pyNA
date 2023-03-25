import numpy as np
from pyNA.src.noise_model.tables import Tables


def calculate_aspl(spl: np.ndarray, f:np.ndarray, tables: Tables):

    """
    
    Arguments
    ---------
    spl : np.ndarray
        _
    f : np.ndarray
        _
    tables : Tables
        _
        
    Outputs
    -------
    aspl : np.ndarray
        _

    """
    
    # Get a-weights
    weights = np.interp(f, tables.levels.aw_f, tables.levels.aw_db)
    
    aspl = 10*np.log10(np.sum(10**((spl + weights)/10.)))

    return aspl

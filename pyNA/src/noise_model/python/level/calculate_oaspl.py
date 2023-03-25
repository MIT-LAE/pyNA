import numpy as np

def calculate_oaspl(spl:np.array):
    
    """


    Parameters
    ----------
    spl : np.array
        _

    """
    
    # Compute 
    oaspl = 10*np.log10(np.sum(10**(spl/10.)))
    
    return oaspl

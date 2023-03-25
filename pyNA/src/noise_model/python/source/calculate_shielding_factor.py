import numpy as np
from pyNA.src.noise_model.tables import Tables


def calculate_shielding_factor(settings: dict, tables: Tables, i_observer:int, i_timestep: int) -> np.ndarray:

    """
    
    Arguments
    ---------
    settings : dict
        pyna_settings
    tables : Tables
        _
    i_observer: index observer
        _
    i_timestep: index time step
        _ 
        
    Outputs
    -------
    shielding : np.array
        _

    """

    if settings['case_name'] == 'nasa_stca_standard' and settings['shielding']:

        if settings['observer_lst'][i_observer] == 'lateral':
            shielding = tables.shielding.lateral[i_timestep]
        
        elif settings['observer_lst'][i_observer] == 'flyover':
            shielding = tables.shielding.flyover[i_timestep]
        
        elif settings['observer_lst'][i_observer] == 'approach':
            shielding = tables.shielding.approach[i_timestep]
        
    else:
        shielding = np.ones(settings['n_frequency_bands'])

    return shielding
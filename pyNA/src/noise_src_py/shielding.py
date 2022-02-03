import pdb
import openmdao
import numpy as np
import openmdao.api as om
from pyNA.src.data import Data
from pyNA.src.settings import Settings


class Shielding(om.ExplicitComponent):
    """
    Compute shielding factors for a trajectory.

    The *Shielding* component requires the following inputs:


    The *Shielding* component computes the following outputs:

    * ``outputs['shield']``:        airframe shielding delta-dB values

    The *Shielding* component has the following options:

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series 
    * ``data``:                     pyna noise data

    """
    def initialize(self):
        # Declare data option
        self.options.declare('settings', types=Settings, desc='pyna settings')
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('data', types=Data)

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings.x_observer_array)[0]

        # Output
        self.add_output('shield', val=np.zeros((n_obs, n_t, settings.N_f)), desc='airframe shielding delta-dB values')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        
        # Load options
        settings = self.options['settings']
        data = self.options['data']

        if settings.case_name in ["NASA STCA Standard", "stca_enginedesign_standard"] and settings.shielding:
                
            for i in np.arange(len(settings.observer_lst)):

                if settings.observer_lst[i] == 'lateral':
                    outputs['shield'][i, :, :] = data.shield_l

                elif settings.observer_lst[i] == 'flyover':
                    outputs['shield'][i, :, :] = data.shield_f

                elif settings.observer_lst[i] == 'approach':
                    outputs['shield'][i, :, :] = data.shield_a

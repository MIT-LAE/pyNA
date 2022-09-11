import pdb
import openmdao
import openmdao.api as om
import numpy as np
from tqdm import tqdm
from pyNA.src.noise_src_py.epnl import epnl
from pyNA.src.noise_src_py.ipnlt import ipnlt
from pyNA.src.noise_src_py.ioaspl import ioaspl

class LevelsInt(om.ExplicitComponent):
    """
    Compute noise levels according to ICAO Annex 16 Volume I: Noise.

    * Sound pressure level (spl)
    * Integrated overall sound pressure level (ioaspl)
    * Integrated pnlt (ipnlt)
    * Effective perceived noise level (epnl)

    The *LevelsInt* component requires the following inputs:

    * ``inputs['oaspl']``:          overall sound pressure level [dB]
    * ``inputs['pnlt']``:           perceived noise level, tone corrected [dB]
    * ``inputs['C']``:              tone-corrections [dB]
    * ``inputs['t_o']``:            observer time [s]

    The *LevelsInt* component computes the following outputs:

    * ``outputs['ioaspl']``:        time-integrated overall sound pressure level [-]
    * ``outputs['ipnlt']``:         time-integrated pnlt [-]
    * ``outputs['epnl']``:          effective perceived noise level [EPNdB]

    The *LevelsInt* component has the following options: 

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series

    """

    def initialize(self):
        self.options.declare('settings', types=dict)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        # Add inputs and outputs
        self.add_input('t_o', val=np.ones((n_obs, n_t)), units='s', desc='observer time [s]')

        if settings['levels_int_metric'] == 'ioaspl':
            self.add_input('oaspl', val=np.ones((n_obs, n_t)), units=None, desc='overall sound pressure level [dB]')
            self.add_output('ioaspl', val=np.ones(n_obs), desc='time-integrated overall sound pressure level [dB]')
        elif settings['levels_int_metric'] == 'ipnlt':
            self.add_input('pnlt', val=np.ones((n_obs, n_t)), units=None, desc='perceived noise level, tone corrected [dB]')
            self.add_output('ipnlt', val=np.ones(n_obs), units=None, desc='time-integrated pnlt [-]')
        elif settings['levels_int_metric'] == 'epnl':
            self.add_input('pnlt', val=np.ones((n_obs, n_t)), units=None, desc='perceived noise level, tone corrected [dB]')
            if settings['epnl_bandshare']:
                self.add_input('C', val=np.ones((n_obs, n_t, settings['n_frequency_bands'])), units=None, desc='tone corrections [dB]')
            self.add_output('epnl', val=np.ones(n_obs), units=None, desc='effective perceived noise level [EPNdB]')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        settings = self.options['settings']

        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        for i in np.arange(n_obs):

            # Compute ioaspl
            if settings['levels_int_metric'] == 'ioaspl':
                outputs['ioaspl'][i] = ioaspl(self, inputs['t_o'][i,:], inputs['oaspl'][i,:])

            # Compute ipnlt
            elif settings['levels_int_metric'] == 'ipnlt':
                outputs['ipnlt'][i] = ipnlt(self, inputs['t_o'][i,:], inputs['pnlt'][i,:])

            # Compute EPNL
            elif settings['levels_int_metric'] == 'epnl':
                if settings['epnl_bandshare']:
                    outputs['epnl'][i] = epnl(self, inputs['t_o'][i,:], inputs['pnlt'][i,:], inputs['C'][i,:])
                else:
                    outputs['epnl'][i] = epnl(self, inputs['t_o'][i,:], inputs['pnlt'][i,:])


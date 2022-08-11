import pdb
import openmdao
import openmdao.api as om
import numpy as np
from pyNA.src.data import Data
from pyNA.src.settings import Settings
from pyNA.src.noise_src_py.spl import spl
from pyNA.src.noise_src_py.oaspl import oaspl
from pyNA.src.noise_src_py.pnlt import pnlt

class Levels(om.ExplicitComponent):
    """
    Compute noise levels according to ICAO Annex 16 Volume I: Noise.

    * Sound pressure level (spl)
    * Overall sound pressure level (oaspl)
    * Perceived noise level, tone-corrected (pnlt)
    * Tone corrections (C)

    The *Levels* component requires the following inputs:

    * ``inputs['rho_0']``:          ambient density [kg/m3]
    * ``inputs['c_0']``:            ambient speed of sound [m/s]
    * ``inputs['msap_prop']``:      mean-square acoustic pressure, propagated to the observer (re. rho_0^2c_0^2) [-]

    The *Levels* component computes the following outputs:

    * ``outputs['spl']``:           sound pressure level [dB]
    * ``outputs['oaspl']``:         overall sound pressure level [dB]
    * ``outputs['pnlt']``:          perceived noise leve, tone corrected [PNdB]
    * ``outputs['C']``:             pnlt tone correction [dB]

    The *LevelsInt* component has the following options: 

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series
    * ``data``:                     pyna noise data

    """

    def initialize(self):
        self.options.declare('settings', types=Settings)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('data', types=Data)

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings.x_observer_array)[0]

        # Add inputs and outputs
        self.add_input('rho_0', val=np.ones(n_t), units='kg/m**3', desc='ambient density [kg/m3]')
        self.add_input('c_0', val=np.ones(n_t), units='m/s', desc='ambient speed of sound [m/s]')
        self.add_input('msap_prop', val=np.ones((n_obs, n_t, settings.N_f)), units=None, desc='mean-square acoustic pressure, propagated to the observer (re. rho_0^2c_0^2) [-]')

        self.add_output('spl', val=np.ones((n_obs, n_t, settings.N_f)), desc='sound pressure level [dB]')
        self.add_output('oaspl', val=np.ones((n_obs, n_t)), desc='overall sound pressure level [dB]')
        
        self.add_output('noy', val=np.ones((n_obs, n_t, settings.N_f)), desc='noy [dB]')
        self.add_output('pnl', val=np.ones((n_obs, n_t)), desc='perceived noise level [PNdB]')
        self.add_output('pnlt', val=np.ones((n_obs, n_t)), desc='perceived noise level, tone corrected [PNdB]')
        self.add_output('c_max', val=np.ones((n_obs, n_t)), desc='maximum tone correction [dB]')
        self.add_output('C', val=np.ones((n_obs, n_t, settings.N_f)), desc='pnlt tone correction [dB]')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        settings = self.options['settings']

        # Number of observers
        n_obs = np.shape(settings.x_observer_array)[0]

        # Extract inputs
        msap_prop = inputs['msap_prop']
        rho_0 = inputs['rho_0']
        c_0 = inputs['c_0']

        for i in np.arange(n_obs):

            # Compute SPL
            outputs['spl'][i, :, :] = spl(self, msap_prop[i, :, :], rho_0, c_0)

            # Compute OASPL
            outputs['oaspl'][i, :] = oaspl(self, outputs['spl'][i, :, :])

            # Compute PNLT and C
            outputs['noy'][i, :, :], outputs['pnl'][i, :], outputs['pnlt'][i, :], outputs['c_max'][i, :], outputs['C'][i, :, :] = pnlt(self, outputs['spl'][i, :, :])
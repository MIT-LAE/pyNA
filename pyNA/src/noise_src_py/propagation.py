import pdb
import openmdao
import numpy as np
import openmdao.api as om
from pyNA.src.data import Data
from pyNA.src.noise_src_py.split_subbands import split_subbands
from noise_src_py.ground_effects import ground_effects
from pyNA.src.noise_src_py.lateral_attenuation import lateral_attenuation


class Propagation(om.ExplicitComponent):
    """
    Computes propagation of mean-square acoustic pressure (msap):

    * Distance-law: R2
    * Characteristic impedance law
    * Atmospheric absorption

    Computes ground reflections and absorption of propagated mean-square acoustic pressure.

    The *Propagation* component requires the following inputs:

    * ``inputs['x']``:              aircraft x-position [m]
    * ``inputs['z']``:              aircraft z-position [m]
    * ``inputs['r']``:              distance source to observer [m]
    * ``inputs['c_bar']``:          average ambient speed of sound between observer and source [m/s]
    * ``inputs['rho_0']``:          ambient density [kg/m3]
    * ``inputs['I_0']``:            characteristic impedance [kg/(m2 s)]
    * ``inputs['beta']``:           elevation angle [deg]
    * ``inputs['msap_source']``:    mean-square acoustic pressure of the source (re. rho_0,^2c_0^2) [-]

    The *Propagation* component computes the following outputs:

    * ``outputs['msap_prop']``:     mean-square acoustic pressure, propagated to the observer (re. rho_0^2c_0^2) [-]

    The *Propagation* component has the following options:

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series 
    * ``data``:                     pyna noise data

    """

    def initialize(self):
        # Declare data option
        self.options.declare('settings', types=dict)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('data', types=Data)

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        # Add inputs
        self.add_input('x', val=np.ones(n_t), units='m', desc='aircraft x-position [m]')
        self.add_input('z', val=np.ones(n_t), units='m', desc='aircraft z-position [m]')
        self.add_input('r', val=np.ones((n_obs, n_t)), units='m', desc='distance source to observer [m]')
        self.add_input('c_bar', val=np.ones((n_obs, n_t)), units='m/s', desc='average ambient speed of sound between observer and source [m/s]')
        self.add_input('rho_0', val=np.ones(n_t), units='kg/m**3', desc='ambient density [kg/m3]')
        self.add_input('I_0', val=np.ones(n_t), units='kg/m**2/s', desc='ambient characteristic impedance [kg/m2/s]')
        self.add_input('beta', val=np.ones((n_obs, n_t)), units='deg', desc='elevation angle [deg]')
        self.add_input('msap_source', val=np.ones((n_obs, n_t, settings['n_frequency_bands'])), desc='mean-square acoustic pressure of the source (re. rho_0,^2c_0^2) [-]')

        self.add_output('msap_prop', val=np.ones((n_obs, n_t, settings['n_frequency_bands'])), desc='mean-square acoustic pressure, propagated to the observer (re. rho_0^2c_0^2) [-]')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        settings = self.options['settings']
        data = self.options['data']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        # Extract inputs
        r = inputs['r']
        x = inputs['x']
        z = inputs['z']
        c_bar = inputs['c_bar']
        rho_0 = inputs['rho_0']
        I_0 = inputs['I_0']
        beta = inputs['beta']
        msap_source = inputs['msap_source']
        I_0_obs = 409.74

        for k in np.arange(n_obs):

            for i in np.arange(n_t):

                # Apply spherical spreading and characteristic impedance effects to the MSAP
                # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
                if settings['direct_propagation']:
                    msap_r = msap_source[k, i, :] * (settings['r_0'] ** 2 / r[k, i] ** 2) * (I_0_obs / I_0[i])
                else:
                    msap_r = msap_source[k, i, :]
    
                # Generate sub-banding
                msap_prop_i = np.zeros(settings['n_frequency_bands'])
                if settings['absorption'] or settings['ground_effects']:

                    msap_sb = split_subbands(settings, msap_r)

                    # Initialize solution vectors
                    if settings['absorption']:
                        # ---------- Apply atmospheric absorption on sub-bands ----------
                        # Compute average absorption factor between observer and source
                        alpha_f = data.abs_f(data.f_sb, z[i])

                        # Compute absorption (convert dB to Np: 1dB is 0.115Np)
                        # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
                        msap_sb = msap_sb * np.exp(-2 * 0.115 * alpha_f * (r[k, i] - settings['r_0']))

                    # ---------- Apply ground effects on sub-bands ----------
                    if settings['ground_effects']:
                        # Empirical lateral attenuation for microphone on sideline
                        if  settings['lateral_attenuation'] and settings['x_observer_array'][k, 1] != 0:
                            # Lateral attenuation factor
                            Lambda = lateral_attenuation(settings, beta[k, i], settings['x_observer_array'][k, :])
                            
                            # Compute elevation angle from center-line observer; set observer z-position to 0
                            r_cl = np.sqrt((x-settings['x_observer_array'][k,1])**2 + 1**2 + z**2)
                            beta_cl = np.arcsin(z/r_cl) * 180. / np.pi

                            # Ground reflection factor for center-line
                            G_cl = ground_effects(settings, data, r_cl, beta_cl, settings['x_observer_array'][k, :], c_bar[k, i], rho_0[i])
                            
                            # Apply ground effects
                            msap_sb = msap_sb * (G_cl * Lambda)

                        # No empirical lateral attenuation or microphone underneath flight path
                        else:
                            # Ground reflection factor
                            G = ground_effects(settings, data, r[k, i], beta[k, i], settings['x_observer_array'][k, :], c_bar[k, i], rho_0[i])
                            
                            # Apply ground effects
                            msap_sb = msap_sb * G

                    # Compute absorbed msap by adding up the msap at all the sub-band frequencies
                    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 22

                    for j in np.arange(settings['n_frequency_bands']):
                        msap_prop_i[j] = np.sum(msap_sb[j*settings['n_frequency_subbands']:(j+1)*settings['n_frequency_subbands']])

                else:
                    msap_prop_i = msap_r

                outputs['msap_prop'][k, i, :] = msap_prop_i.clip(min=1e-99)
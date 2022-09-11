from multiprocessing.sharedctypes import Value
import pdb
import openmdao
import openmdao.api as om
import numpy as np


class Geometry(om.ExplicitComponent):
    """
    Compute geometrical parameters and ambient parameters along the trajectory.

    The *Geometry* component requires the following inputs:

    * ``inputs['x']``:              aircraft x-position [m]
    * ``inputs['y']``:              aircraft y-position [m]
    * ``inputs['z']``:              aircraft z-position [m]
    * ``inputs['alpha']``:          aircraft angle of attack [deg]
    * ``inputs['gamma']``:          aircraft climb angle [deg]
    * ``inputs['c_0']``:            ambient speed of sound [m/s]
    * ``inputs['T_0']``:            ambient temperature [K]
    * ``inputs['t_s']``:            source time [s]

    The *Geometry* component computes the following outputs:

    * ``outputs['r']``:             distance source to observer [m]
    * ``outputs['theta']``:         polar directivity angle [deg]
    * ``outputs['phi']``:           azimuthal directivity angle [deg]
    * ``outputs['beta']``:          elevation angle [deg]
    * ``outputs['t_o']``:           observer time [s]
    * ``outputs['c_bar']``:         average speed of sound between source and observer [m/s]

    The *Geometry* component has the following options :

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series
    * ``mode``:                     mode for geometry calculations: "trajectory" / "distribution"

    """

    def initialize(self):
        # Declare data option
        self.options.declare('settings', types=dict, desc='noise settings')
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('mode', types=str, desc='mode for geometry calculations')

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']
        
        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        # Add inputs and outputs
        self.add_input('x', val=np.ones(n_t), units='m', desc='aircraft x-position [m]')
        self.add_input('y', val=np.ones(n_t), units='m', desc='aircraft y-position [m]')
        self.add_input('z', val=np.ones(n_t), units='m', desc='aircraft z-position [m]')
        self.add_input('alpha', val=np.ones(n_t), units='deg', desc='aircraft angle of attack [deg]')
        self.add_input('gamma', val=np.ones(n_t), units='deg', desc='aircraft climb angle [deg]')
        self.add_input('c_0', val=np.ones(n_t), units='m/s', desc='ambient speed of sound [m/s]')
        self.add_input('T_0', val=np.ones(n_t), units='K', desc='ambient temperature [K]')
        self.add_input('t_s', val=np.ones(n_t), units='s', desc='source time [s]')

        self.add_output('r', val=np.ones((n_obs, n_t)), units='m', desc='distance source to observer [m]')
        self.add_output('theta', val=np.ones((n_obs, n_t)), units='deg', desc='polar directivity angle [deg]')
        self.add_output('phi', val=np.ones((n_obs, n_t)), units='deg', desc='azimuthal directivity angle [deg]')
        self.add_output('beta', val=np.ones((n_obs, n_t)), units='deg', desc='elevation angle [deg]')
        self.add_output('t_o', val=np.ones((n_obs, n_t)), units='s', desc='observer time [s]')
        self.add_output('c_bar', val=np.ones((n_obs, n_t)), units='m/s', desc='average speed of sound between source and observer [m/s]')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']
        mode = self.options['mode']

        # Number of observers
        n_obs = np.shape(settings['x_observer_array'])[0]

        # Extract inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        alpha = inputs['alpha']
        gamma = inputs['gamma']
        t_s = inputs['t_s']
        c_0 = inputs['c_0']
        T_0 = inputs['T_0']

        if mode == 'trajectory':
            # Iterate over observers
            for i in np.arange(n_obs):

                # Geometry calculations
                # Compute body angles (psi_B, theta_B, phi_B): angle of body w.r.t. horizontal
                theta_B = alpha + gamma
                phi_B = np.zeros(alpha.shape)
                psi_B = np.zeros(alpha.shape)

                # Compute the relative observer-source position vector i.e. difference between observer and ac coordinate
                # Note: add 4 meters to the alitude of the aircraft (for engine height)
                r_0 =  settings['x_observer_array'][i,0] - x
                r_1 =  settings['x_observer_array'][i,1] - y
                r_2 = -settings['x_observer_array'][i,2] + (z + 4)

                # Compute the distance of the observer-source vector
                R = np.sqrt(r_0 ** 2 + r_1 ** 2 + r_2 ** 2)
                outputs['r'][i,:] = R

                # Normalize the distance vector
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 17
                n_vcr_a_0 = r_0 / R
                n_vcr_a_1 = r_1 / R
                n_vcr_a_2 = r_2 / R

                # Define elevation angle
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 21
                outputs['beta'][i,:] = 180. / np.pi * np.arcsin(n_vcr_a_2)

                # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 22-25
                cth  = np.cos(np.pi / 180. * theta_B)
                sth  = np.sin(np.pi / 180. * theta_B)
                cphi = np.cos(np.pi / 180. * phi_B)
                sphi = np.sin(np.pi / 180. * phi_B)
                cpsi = np.cos(np.pi / 180. * psi_B)
                spsi = np.sin(np.pi / 180. * psi_B)

                n_vcr_s_0 = cth * cpsi * n_vcr_a_0 + cth * spsi * n_vcr_a_1 - sth * n_vcr_a_2
                n_vcr_s_1 = (-spsi * cphi + sphi * sth * cpsi) * n_vcr_a_0 + ( cphi * cpsi + sphi * sth * spsi) * n_vcr_a_1 + sphi * cth * n_vcr_a_2
                n_vcr_s_2 = (spsi * sphi + cphi * sth * cpsi) * n_vcr_a_0 + ( -sphi * cpsi + cphi * sth * spsi) * n_vcr_a_1 + cphi * cth * n_vcr_a_2

                # Compute polar directivity angle
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 26
                theta = 180. / np.pi * np.arccos(n_vcr_s_0)
                outputs['theta'][i,:] = theta

                # Compute azimuthal directivity angle
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 27
                phi = -180. / np.pi * np.arctan2(n_vcr_s_1, n_vcr_s_2)
                if settings['case_name'] in ["nasa_stca_standard", "stca_enginedesign_standard"]:
                    outputs['phi'][i, :] = np.zeros(n_t)
                else:
                    phi[phi==-180.] = np.zeros(n_t)[phi==-180.]
                    outputs['phi'][i,:] = phi

                # Compute average speed of sound between source and observer
                n_intermediate = 11
                dz = z / n_intermediate
                c_bar = c_0
                for k in np.arange(1, n_intermediate):
                    T_im = T_0 - k * dz * (-0.0065)
                    c_im = np.sqrt(1.4 * 287. * T_im)
                    c_bar = (k) / (k + 1) * c_bar + c_im / (k + 1)
                outputs['c_bar'][i,:] = c_bar

                # Compute observed time
                # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 20
                outputs['t_o'][i,:] = t_s + outputs['r'][i,:] / outputs['c_bar'][i,:]

        elif mode == 'distribution':
            outputs['r'] = 0.3048*np.ones((1, 19))
            outputs['theta'] = np.reshape(np.linspace(0, 180, 19), (1,19))
            outputs['phi'] = np.zeros((1,19))
            outputs['beta'] = np.zeros((1,19))
            outputs['t_o'] = np.zeros((1,19))
            outputs['c_bar'] = np.zeros((1,19))

        else:
            raise ValueError('Invalid mode specified. Specify: "time_series" / "distribution".')

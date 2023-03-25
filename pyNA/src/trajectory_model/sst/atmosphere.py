import pdb
import numpy as np
import openmdao
import openmdao.api as om


class Atmosphere(om.ExplicitComponent):
    """
    Compute ambient parameters along the trajectory.

    The *Atmosphere* component requires the following inputs:

    * ``inputs['z']``:              aircraft z-position [m]

    The *Atmosphere* component computes the following outputs:

    * ``outputs['P_0']``:           ambient pressure [Pa]
    * ``outputs['rho_0']``:         ambient density [kg/m3]
    * ``outputs['T_0']``:           ambient temperature [K]
    * ``outputs['c_0']``:           ambient speed of sound [m/s]
    * ``outputs['c_bar']``:         average ambient speed of sound between observer and source [m/s]
    * ``outputs['mu_0']``:          ambient dynamic viscosity [kg/ms]
    * ``outputs['k_0']``:           ambient thermal conductivity [W/mK]
    * ``outputs['I_0']``:           ambient characteristic impedance [kg/m2/s]

    """

    def initialize(self):
        # Declare data option
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('settings')

        self.sl = dict()
        self.sl['g'] = 9.80665
        self.sl['R'] = 287.05
        self.sl['T_0'] = 288.15
        self.sl['c_0'] = 340.294
        self.sl['P_0'] = 101325.
        self.sl['rho_0'] = 1.225
        self.sl['mu_0'] = 1.7894e-5
        self.sl['k_0'] = 25.5e-3
        self.sl['lapse_0'] = 0.0065
        self.sl['gamma'] = 1.4
        self.sl['rh_0'] = 70.
        self.sl['I_0'] = 409.74

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('z', val=np.ones(nn), units='m', desc='aircraft z-position [m]')
        
        self.add_output('P_0', val=np.ones(nn), units='Pa', desc='ambient pressure')
        self.add_output('rho_0', val=np.ones(nn), units='kg/m**3', desc='ambient density')
        self.add_output('drho_0_dz', val=1.*np.ones(nn), units='kg/m**4', desc='change of density with altitude')
        self.add_output('T_0', val=np.ones(nn), units='K', desc='ambient temperature')
        self.add_output('c_0', val=np.ones(nn), units='m/s', desc='ambient speed of sound')
        self.add_output('mu_0', val=np.ones(nn), units='kg/m/s', desc='ambient dynamic viscosity')
        self.add_output('I_0', val=np.ones(nn), units='kg/m**2/s', desc='ambient characteristic impedance')
        self.add_output('rh', val=np.ones(nn), units=None, desc='Relative humidity')

    def setup_partials(self):
        """
        Declare partials.
        """
        
        nn = self.options['num_nodes']

        ar = np.arange(nn)
        self.declare_partials('T_0', 'z', rows=ar, cols=ar)
        self.declare_partials('P_0', 'z', rows=ar, cols=ar)
        self.declare_partials('rho_0', 'z', rows=ar, cols=ar)
        self.declare_partials('drho_0_dz', 'z', rows=ar, cols=ar)
        self.declare_partials('c_0', 'z', rows=ar, cols=ar)
        self.declare_partials('mu_0', 'z', rows=ar, cols=ar)
        self.declare_partials('I_0', 'z', rows=ar, cols=ar)
        self.declare_partials('rh', 'z', rows=ar, cols=ar)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        nn = self.options['num_nodes']
        settings = self.options['settings']

        if settings['atmosphere_mode'] == 'stratified':
            z = inputs['z']
        elif settings['atmosphere_mode'] == 'sealevel':
            z = np.zeros(nn,)
        else:
            raise ValueError("Atmosphere mode '" + settings['atmosphere_mode'] + "' is invalid.")

        # Temperature, pressure, density and speed of sound
        T_isa = self.sl['T_0'] - z * self.sl['lapse_0']  # Temperature without dT_isa
        outputs['T_0'] = self.sl['T_0'] + settings['atmosphere_dT'] - z * self.sl['lapse_0']  # Temperature
        outputs['P_0'] = self.sl['P_0'] * (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'])  # Pressure
        outputs['rho_0'] = outputs['P_0'] / outputs['T_0'] / self.sl['R']  # Density
        if settings['atmosphere_mode'] == 'stratified':
            dT_0_dz = -self.sl['lapse_0']
            dP_0_dz =  self.sl['P_0'] * ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R']) * (-self.sl['lapse_0'] / self.sl['T_0']) * (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'] - 1)
            outputs['drho_0_dz'] = 1/self.sl['R'] * (dP_0_dz*outputs['T_0'] - outputs['P_0']*dT_0_dz)/outputs['T_0']**2
        else:
            outputs['drho_0_dz'] = np.zeros(nn,)
        outputs['c_0'] = np.sqrt(self.sl['gamma'] * self.sl['R'] * outputs['T_0'])  # Speed of sound

        # Dynamic viscosity
        # Source: Zorumski report 1982 part 1. Chapter 2.1 Equation 11
        outputs['mu_0'] = self.sl['mu_0'] * (1.38313 * (outputs['T_0'] / self.sl['T_0']) ** 1.5) / ( outputs['T_0'] / self.sl['T_0'] + 0.38313)

        # Characteristic impedance
        # Source: Zorumski report 1982 part 1. Chapter 2.1 Equation 13
        outputs['I_0'] = self.sl['rho_0'] * self.sl['c_0'] * outputs['P_0'] / self.sl['P_0'] * (self.sl['T_0'] / outputs['T_0']) ** 0.5

        # Relative humidity 
        # Source: Berton, NASA STCA Release Package 2018. Note: relative humidity at sea-level in standard day is 70%.
        outputs['rh'] = self.sl['rh_0'] - 0.012467191601049869 * z

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        
        nn = self.options['num_nodes']
        settings = self.options['settings']

        z = inputs['z']

        if settings['atmosphere_mode'] == 'stratified':
            # Temperature, pressure, density and speed of sound
            T_isa = self.sl['T_0'] - z * self.sl['lapse_0']  # Temperature without dT_isa
            T_0 = self.sl['T_0'] + settings['atmosphere_dT'] - z * self.sl['lapse_0']  # Temperature
            P_0 = self.sl['P_0'] * (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'])  # Pressure
            c_0 = np.sqrt(self.sl['gamma'] * self.sl['R'] * T_0)  # Speed of sound

            # Calculate partials
            partials['T_0', 'z'] = -self.sl['lapse_0']
            partials['P_0', 'z'] = self.sl['P_0'] * ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R']) * (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'] - 1) *  (-self.sl['lapse_0'] / self.sl['T_0'])
            partials['rho_0', 'z'] = 1/self.sl['R'] * (partials['P_0', 'z']*T_0 - P_0*partials['T_0', 'z'])/T_0**2
            partials['c_0', 'z'] = 1/2/np.sqrt(self.sl['gamma'] * self.sl['R'] * T_0) * self.sl['gamma'] * self.sl['R']*(-self.sl['lapse_0'])
            
            dT_0_dz = -self.sl['lapse_0']
            dP_0_dz =  self.sl['P_0'] * ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R']) * \
                    (-self.sl['lapse_0'] / self.sl['T_0']) * \
                    (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'] - 1)         
            dT_0_dz2 = 0
            dP_0_dz2 = self.sl['P_0'] * ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R']) * \
                    (-self.sl['lapse_0'] / self.sl['T_0']) * \
                    ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'] - 1)*\
                    (T_isa / self.sl['T_0']) ** ( self.sl['g'] / self.sl['lapse_0'] / self.sl['R'] - 2) * \
                    (-self.sl['lapse_0'] / self.sl['T_0'])
            dnum_dz = dP_0_dz2*T_0 + dP_0_dz*dT_0_dz - dP_0_dz*dT_0_dz
            partials['drho_0_dz', 'z'] = 1/self.sl['R'] * (dnum_dz*T_0**2 - (dP_0_dz*T_0 - P_0*dT_0_dz)*2*T_0*dT_0_dz)/T_0**4

            dmuN_dz = 1.38313 * (1.5*T_0**0.5*(-self.sl['lapse_0'])/self.sl['T_0']**1.5)
            dmuD_dz = ( -self.sl['lapse_0'] / self.sl['T_0'] )
            partials['mu_0', 'z'] = self.sl['mu_0'] * (dmuN_dz*( T_0 / self.sl['T_0'] + 0.38313) -  (1.38313 * (T_0 / self.sl['T_0']) ** 1.5)*dmuD_dz)/( T_0 / self.sl['T_0'] + 0.38313)**2

            partials['I_0', 'z'] = (self.sl['rho_0'] * self.sl['c_0']/ self.sl['P_0'] * self.sl['T_0']**0.5) * (dP_0_dz*T_0**0.5 - P_0*0.5*T_0**(-0.5)*dT_0_dz)/T_0

            partials['rh', 'z'] = -0.012467191601049869*np.ones(nn,)

        elif settings['atmosphere_mode'] == 'sealevel':
            pass

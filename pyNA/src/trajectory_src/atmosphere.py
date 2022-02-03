import pdb
import openmdao
import openmdao.api as om
from pyNA.src.settings import Settings
import numpy as np


class Atmosphere(om.ExplicitComponent):
    """
    Compute ambient parameters along the trajectory.

    The *Atmosphere* component requires the following inputs:

    * ``inputs['z']``:              aircraft z-position [m]

    The *Atmosphere* component computes the following outputs:

    * ``outputs['p_0']``:           ambient pressure [Pa]
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
        self.options.declare('settings', types=Settings)
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')

        # Constants
        self.varAtm = dict()
        self.varAtm['g'] = 9.80665
        self.varAtm['R'] = 287.05
        self.varAtm['T0'] = 288.15
        self.varAtm['c0'] = 340.294
        self.varAtm['p0'] = 101325.
        self.varAtm['rho0'] = 1.225
        self.varAtm['mu0'] = 1.7894e-5
        self.varAtm['k0'] = 25.5e-3
        self.varAtm['lapse0'] = 0.0065
        self.varAtm['gamma'] = 1.4
        self.varAtm['rh0'] = 70.

    def setup(self):

        # Load options
        nn = self.options['num_nodes']

        # Add inputs and outputs
        self.add_input('z', val=np.ones(nn), units='m', desc='aircraft z-position [m]')
        
        self.add_output('p_0', val=np.ones(nn), units='Pa', desc='ambient pressure [Pa]')
        self.add_output('rho_0', val=np.ones(nn), units='kg/m**3', desc='ambient density [kg/m3]')
        self.add_output('drho_0_dz', val=1.*np.ones(nn), units='kg/m**4', desc='change of density with altitude [kg/m4]')
        self.add_output('T_0', val=np.ones(nn), units='K', desc='ambient temperature [K]')
        self.add_output('c_0', val=np.ones(nn), units='m/s', desc='ambient speed of sound [m/s]')
        self.add_output('mu_0', val=np.ones(nn), units='kg/m/s', desc='ambient dynamic viscosity [kg/ms]')
        self.add_output('I_0', val=np.ones(nn), units='kg/m**2/s', desc='ambient characteristic impedance [kg/m2/s]')
        self.add_output('rh', val=np.ones(nn), units=None, desc='Relative humidity [%]')

    def setup_partials(self):
        """
        Declare partials.
        """
        # Load options
        nn = self.options['num_nodes']

        ar = np.arange(nn)
        self.declare_partials('T_0', 'z', rows=ar, cols=ar)
        self.declare_partials('p_0', 'z', rows=ar, cols=ar)
        self.declare_partials('rho_0', 'z', rows=ar, cols=ar)
        self.declare_partials('drho_0_dz', 'z', rows=ar, cols=ar)
        self.declare_partials('c_0', 'z', rows=ar, cols=ar)
        self.declare_partials('mu_0', 'z', rows=ar, cols=ar)
        self.declare_partials('I_0', 'z', rows=ar, cols=ar)
        self.declare_partials('rh', 'z', rows=ar, cols=ar)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        settings = self.options['settings']

        # Extract inputs
        z = inputs['z']
        
        # Temperature, pressure, density and speed of sound
        T_isa = self.varAtm['T0'] - z * self.varAtm['lapse0']  # Temperature without dT_isa
        outputs['T_0'] = self.varAtm['T0'] + settings.dT - z * self.varAtm['lapse0']  # Temperature
        outputs['p_0'] = self.varAtm['p0'] * (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'])  # Pressure
        outputs['rho_0'] = outputs['p_0'] / outputs['T_0'] / self.varAtm['R']  # Density

        dT_0_dz = -self.varAtm['lapse0']
        dp_0_dz =  self.varAtm['p0'] * ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R']) * (-self.varAtm['lapse0'] / self.varAtm['T0']) * (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'] - 1)
        outputs['drho_0_dz'] = 1/self.varAtm['R'] * (dp_0_dz*outputs['T_0'] - outputs['p_0']*dT_0_dz)/outputs['T_0']**2
        outputs['c_0'] = np.sqrt(self.varAtm['gamma'] * self.varAtm['R'] * outputs['T_0'])  # Speed of sound

        # Dynamic viscosity
        # Source: Zorumski report 1982 part 1. Chapter 2.1 Equation 11
        outputs['mu_0'] = self.varAtm['mu0'] * (1.38313 * (outputs['T_0'] / self.varAtm['T0']) ** 1.5) / ( outputs['T_0'] / self.varAtm['T0'] + 0.38313)

        # Characteristic impedance
        # Source: Zorumski report 1982 part 1. Chapter 2.1 Equation 13
        outputs['I_0'] = self.varAtm['rho0'] * self.varAtm['c0'] * outputs['p_0'] / self.varAtm['p0'] * (self.varAtm['T0'] / outputs['T_0']) ** 0.5

        # Relative humidity 
        # Source: Berton, NASA STCA Release Package 2018. Note: relative humidity at sea-level in standard day is 70%.
        outputs['rh'] = -0.012467191601049869 * z + self.varAtm['rh0']

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        nn = self.options['num_nodes']
        settings = self.options['settings']

        # Load inputs
        z = inputs['z']

        # Temperature, pressure, density and speed of sound
        T_isa = self.varAtm['T0'] - z * self.varAtm['lapse0']  # Temperature without dT_isa
        T_0 = self.varAtm['T0'] + settings.dT - z * self.varAtm['lapse0']  # Temperature
        p_0 = self.varAtm['p0'] * (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'])  # Pressure
        c_0 = np.sqrt(self.varAtm['gamma'] * self.varAtm['R'] * T_0)  # Speed of sound

        # Calculate partials
        partials['T_0', 'z'] = -self.varAtm['lapse0']
        partials['p_0', 'z'] = self.varAtm['p0'] * ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R']) * (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'] - 1) *  (-self.varAtm['lapse0'] / self.varAtm['T0'])
        partials['rho_0', 'z'] = 1/self.varAtm['R'] * (partials['p_0', 'z']*T_0 - p_0*partials['T_0', 'z'])/T_0**2
        partials['c_0', 'z'] = 1/2/np.sqrt(self.varAtm['gamma'] * self.varAtm['R'] * T_0) * self.varAtm['gamma'] * self.varAtm['R']*(-self.varAtm['lapse0'])
        
        dT_0_dz = -self.varAtm['lapse0']
        dp_0_dz =  self.varAtm['p0'] * ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R']) * \
                 (-self.varAtm['lapse0'] / self.varAtm['T0']) * \
                  (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'] - 1)         
        dT_0_dz2 = 0
        dp_0_dz2 = self.varAtm['p0'] * ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R']) * \
                 (-self.varAtm['lapse0'] / self.varAtm['T0']) * \
                 ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'] - 1)*\
                 (T_isa / self.varAtm['T0']) ** ( self.varAtm['g'] / self.varAtm['lapse0'] / self.varAtm['R'] - 2) * \
                 (-self.varAtm['lapse0'] / self.varAtm['T0'])
        dnum_dz = dp_0_dz2*T_0 + dp_0_dz*dT_0_dz - dp_0_dz*dT_0_dz
        partials['drho_0_dz', 'z'] = 1/self.varAtm['R'] * (dnum_dz*T_0**2 - (dp_0_dz*T_0 - p_0*dT_0_dz)*2*T_0*dT_0_dz)/T_0**4

        dmuN_dz = 1.38313 * (1.5*T_0**0.5*(-self.varAtm['lapse0'])/self.varAtm['T0']**1.5)
        dmuD_dz = ( -self.varAtm['lapse0'] / self.varAtm['T0'] )
        partials['mu_0', 'z'] = self.varAtm['mu0'] * (dmuN_dz*( T_0 / self.varAtm['T0'] + 0.38313) -  (1.38313 * (T_0 / self.varAtm['T0']) ** 1.5)*dmuD_dz)/( T_0 / self.varAtm['T0'] + 0.38313)**2

        partials['I_0', 'z'] = (self.varAtm['rho0'] * self.varAtm['c0']/ self.varAtm['p0'] * self.varAtm['T0']**0.5) * (dp_0_dz*T_0**0.5 - p_0*0.5*T_0**(-0.5)*dT_0_dz)/T_0

        partials['rh', 'z'] = -0.012467191601049869*np.ones(nn,)
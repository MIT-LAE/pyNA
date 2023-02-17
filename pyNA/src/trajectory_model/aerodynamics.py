import pdb
import numpy as np
import openmdao
import openmdao.api as om
from pyNA.src.airframe import Airframe


class Aerodynamics(om.ExplicitComponent):
    """ 
    Computes aerodynamic forces and Mach number along the trajectory.

    The *Aerodynamics* component requires the following inputs:

    * ``inputs['c_l']``:     aircraft lift coefficient [-]
    * ``inputs['c_d']``:     aircraft drag coefficient [-]
    * ``inputs['v']``:       aircraft velocity [m/s]

    If stratified atmosphere, the *Aerodynamics* component requires additional inputs:

    * ``inputs['rho_0']``:   ambient density [kg/m3]
    * ``inputs['c_0']``:     ambient speed of sound [m/s]


    The *Aerodynamics* component computes the following outputs:

    * ``outputs['L']``:      aircraft lift [N]
    * ``outputs['D']``:      aircraft drag [N]
    * ``outputs['M_0']``:    ambient Mach number [-]

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('phase', types=str)
        self.options.declare('airframe', types=Airframe)
        self.options.declare('atmosphere_type', types=str)
        self.options.declare('sealevel_atmosphere', types=dict, default=dict())

    def setup(self):
        # Load options
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('c_l', shape=(nn,), desc='lift coefficient', units=None)
        self.add_input('c_d', shape=(nn,), desc='lift coefficient', units=None)
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')

        if self.options['atmosphere_type'] == 'stratified':
            self.add_input(name='rho_0', shape=(nn,), desc='atmospheric density', units='kg/m**3')	
            self.add_input(name='c_0', shape=(nn,), desc='atmospheric speed of sound', units='m/s')
        
        # Outputs
        self.add_output(name='L', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='D', shape=(nn,), desc='aerodynamic drag force', units='N')
        self.add_output(name='M_0', shape=(nn,), desc='Mach number', units=None)

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='L', wrt='c_l', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='v', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='c_l', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='c_d', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='v', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='M_0', wrt='v', rows=ar, cols=ar, val=1.0)
        
        if self.options['atmosphere_type'] == 'stratified':
            self.declare_partials(of='L', wrt='rho_0', dependent=True, rows=ar, cols=ar)
            self.declare_partials(of='D', wrt='rho_0', dependent=True, rows=ar, cols=ar)
            self.declare_partials(of='M_0', wrt='c_0', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        airframe = self.options['airframe']
        phase_name = self.options['phase']

        # Atmospheric properties
        if self.options['atmosphere_type'] == 'stratified':
            rho_0 = inputs['rho_0']
            c_0 = inputs['c_0']
        elif self.options['atmosphere_type'] == 'sealevel':
            rho_0 = self.options['sealevel_atmosphere']['rho_0']
            c_0 = self.options['sealevel_atmosphere']['c_0']

        # Dynamic pressure
        q = 0.5 * rho_0 * inputs['v'] ** 2

        # Mach number
        outputs['M_0'] = inputs['v'] / c_0

        # Forces
        outputs['L'] = q * airframe.af_S_w * inputs['c_l']

        constant_LD = False
        if constant_LD:
            L_D = 6.718101501415649 
            outputs['D'] = q * airframe.af_S_w * inputs['c_l']/L_D
        else:
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                outputs['D'] = q * airframe.af_S_w * (inputs['c_d'] + airframe.c_d_g)
            elif phase_name in {'vnrs', 'cutback'}:
                outputs['D'] = q * airframe.af_S_w * inputs['c_d']

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        airframe = self.options['airframe']
        phase_name = self.options['phase']

        # Atmospheric properties
        if self.options['atmosphere_type'] == 'stratified':
            rho_0 = inputs['rho_0']
            c_0 = inputs['c_0']
        elif self.options['atmosphere_type'] == 'sealevel':
            rho_0 = self.options['sealevel_atmosphere']['rho_0']
            c_0 = self.options['sealevel_atmosphere']['c_0']

        # Compute dynamic pressure
        q = 0.5 * rho_0 * inputs['v'] ** 2

        partials['L', 'c_l'] = q * airframe.af_S_w
        partials['L', 'v'] = inputs['c_l'] * rho_0 * inputs['v'] * airframe.af_S_w
        if self.options['atmosphere_type'] == 'stratified':
            partials['L', 'rho_0'] = inputs['c_l'] * 1/2. * inputs['v']**2 * airframe.af_S_w

        constant_LD = False
        if constant_LD:
            L_D = 6.718101501415649
            partials['D', 'c_d'] = 0.
            partials['D', 'c_l'] = q * airframe.af_S_w * 1/L_D
            if self.options['atmosphere_type'] == 'stratified':
                partials['D', 'rho_0'] = 1/2 * inputs['v']**2 * airframe.af_S_w * inputs['c_l']/L_D
            partials['D', 'v'] = rho_0 * inputs['v'] * airframe.af_S_w * inputs['c_l']/L_D
        else:
            partials['D', 'c_d'] = q * airframe.af_S_w
            partials['D', 'c_l'] = 0.
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                if self.options['atmosphere_type'] == 'stratified':
                    partials['D', 'rho_0'] = (inputs['c_d'] + airframe.c_d_g) * 1/2. * inputs['v']**2 * airframe.af_S_w
                partials['D', 'v'] = (inputs['c_d'] + airframe.c_d_g) * rho_0 * inputs['v'] * airframe.af_S_w
            elif phase_name in {'vnrs', 'cutback'}:
                if self.options['atmosphere_type'] == 'stratified':
                    partials['D', 'rho_0'] = inputs['c_d'] * 1/2. * inputs['v']**2 * airframe.af_S_w
                partials['D', 'v'] = inputs['c_d'] * rho_0 * inputs['v'] * airframe.af_S_w 

        partials['M_0', 'v'] = 1.0 / c_0

        if self.options['atmosphere_type'] == 'stratified':
            partials['M_0', 'c_0'] = -inputs['v'] / c_0 ** 2

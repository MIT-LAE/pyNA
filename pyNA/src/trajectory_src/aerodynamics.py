import pdb
import numpy as np
import openmdao
import openmdao.api as om
from pyNA.src.aircraft import Aircraft


class Aerodynamics(om.ExplicitComponent):
    """
    Computes aerodynamic forces and Mach number along the trajectory.

    The *Aerodynamics* component requires the following inputs:

    * ``inputs['c_l']``:     aircraft lift coefficient [-]
    * ``inputs['c_d']``:     aircraft drag coefficient [-]
    * ``inputs['rho_0']``:   ambient density [kg/m3]
    * ``inputs['c_0']``:     ambient speed of sound [m/s]
    * ``inputs['v']``:       aircraft velocity [m/s]

    The *Aerodynamics* component computes the following outputs:

    * ``outputs['q']``:      ambient dynamic pressure [Pa]
    * ``outputs['L']``:      aircraft lift [N]
    * ``outputs['D']``:      aircraft drag [N]
    * ``outputs['M_0']``:    ambient Mach number [-]

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('ac', types=Aircraft)
        self.options.declare('phase', types=str)

    def setup(self):
        # Load options
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('c_l', shape=(nn,), desc='lift coefficient', units=None)
        self.add_input('c_d', shape=(nn,), desc='lift coefficient', units=None)
        self.add_input(name='rho_0', shape=(nn,), desc='atmospheric density', units='kg/m**3')	
        self.add_input(name='c_0', shape=(nn,), desc='atmospheric speed of sound', units='m/s')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')

        # Outputs
        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')
        self.add_output(name='L', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='D', shape=(nn,), desc='aerodynamic drag force', units='N')
        self.add_output(name='M_0', shape=(nn,), desc='Mach number', units=None)

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho_0', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='v', rows=ar, cols=ar)

        self.declare_partials(of='L', wrt='c_l', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='rho_0', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='v', dependent=True, rows=ar, cols=ar)

        self.declare_partials(of='D', wrt='c_l', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='c_d', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='rho_0', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='v', dependent=True, rows=ar, cols=ar)

        self.declare_partials(of='M_0', wrt='v', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='M_0', wrt='c_0', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        ac = self.options['ac']
        phase_name = self.options['phase']

        # Dynamic pressure
        outputs['q'] = 0.5 * inputs['rho_0'] * inputs['v'] ** 2

        # Mach number
        outputs['M_0'] = inputs['v'] / inputs['c_0']

        # Forces
        outputs['L'] = outputs['q'] * ac.af_S_w * inputs['c_l']

        constant_LD = False
        if constant_LD:
            L_D = 6.718101501415649*2 
            outputs['D'] = outputs['q'] * ac.af_S_w * inputs['c_l']/L_D
        else:
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                outputs['D'] = outputs['q'] * ac.af_S_w * (inputs['c_d'] + ac.c_d_g)
            elif phase_name in {'vnrs', 'cutback'}:
                outputs['D'] = outputs['q'] * ac.af_S_w * inputs['c_d']

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        ac = self.options['ac']
        phase_name = self.options['phase']

        # Compute dynamic pressure
        q = 0.5 * inputs['rho_0'] * inputs['v'] ** 2

        partials['q', 'rho_0'] = 0.5 * inputs['v'] ** 2
        partials['q', 'v'] = inputs['rho_0'] * inputs['v']

        partials['L', 'c_l'] = q * ac.af_S_w
        partials['L', 'rho_0'] = inputs['c_l'] * 1/2. * inputs['v']**2 * ac.af_S_w
        partials['L', 'v'] = inputs['c_l'] * inputs['rho_0'] * inputs['v'] * ac.af_S_w

        constant_LD = False
        if constant_LD:
            L_D = 6.718101501415649*2
            partials['D', 'c_d'] = 0.
            partials['D', 'c_l'] = q * ac.af_S_w * 1/L_D
            partials['D', 'rho_0'] = 1/2 * inputs['v']**2 * ac.af_S_w * inputs['c_l']/L_D
            partials['D', 'v'] = inputs['rho_0'] * inputs['v'] * ac.af_S_w * inputs['c_l']/L_D
        else:
            partials['D', 'c_d'] = q * ac.af_S_w
            partials['D', 'c_l'] = 0.
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                partials['D', 'rho_0'] = (inputs['c_d'] + ac.c_d_g) * 1/2. * inputs['v']**2 * ac.af_S_w
                partials['D', 'v'] = (inputs['c_d'] + ac.c_d_g) * inputs['rho_0'] * inputs['v'] * ac.af_S_w
            elif phase_name in {'vnrs', 'cutback'}:
                partials['D', 'rho_0'] = inputs['c_d'] * 1/2. * inputs['v']**2 * ac.af_S_w
                partials['D', 'v'] = inputs['c_d'] * inputs['rho_0'] * inputs['v'] * ac.af_S_w 

        partials['M_0', 'v'] = 1.0 / inputs['c_0']
        partials['M_0', 'c_0'] = -inputs['v'] / inputs['c_0'] ** 2

import pdb
import numpy as np
import openmdao.api as om
from pyNA.src.airframe import Airframe
from openmdao.components.interp_util.interp import TABLE_METHODS


class CLCD(om.MetaModelStructuredComp):
    """
    Computes aerodynamic lift and drag coefficients from aerodynamics data tables.

    The *CLCD* component requires the following inputs:

    * ``inputs['alpha']``:              aircraft angle of attack [deg]
    * ``inputs['theta_flaps']``:        aircraft flap deflection angle [deg]
    * ``inputs['theta_slats']``:        aircraft slat deflection angle [deg]

    The *CLCD* component computes the following outputs:

    * ``outputs['c_l']``:                aircraft lift coefficient [-]
    * ``outputs['c_l_max']``:            maximum aircraft lift coefficient [-]
    * ``outputs['c_d']``:                aircraft drag coefficient [-]
    
    """

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('vec_size', types=int, default=1, desc='Number of points to evaluate at once.')
        self.options.declare('extrapolate', types=bool, default=False, desc='Sets whether extrapolation should be performed when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False, desc='Sets whether gradients with respect to output training data should be computed.')
        self.options.declare('method', values=TABLE_METHODS, default='scipy_cubic', desc='Spline interpolation method to use for all outputs.')
        self.options.declare('airframe', types=Airframe)

    def setup(self):
        # Load options
        nn = self.options['vec_size']
        airframe = self.options['airframe']

        self.add_input('alpha', val=np.ones(nn), units='deg', training_data=airframe.aero['alpha'])
        self.add_input('theta_flaps', val=np.ones(nn), units='deg', training_data=airframe.aero['theta_flaps'])
        self.add_input('theta_slats', val=np.ones(nn), units='deg', training_data=airframe.aero['theta_slats'])

        self.add_output('c_l', val=np.ones(nn), units=None, training_data=airframe.aero['c_l'])
        self.add_output('c_l_max', val=np.ones(nn), units=None, training_data=airframe.aero['c_l_max'])
        self.add_output('c_d', val=np.ones(nn), units=None, training_data=airframe.aero['c_d'])
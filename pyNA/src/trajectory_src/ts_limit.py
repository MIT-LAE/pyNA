import pdb
import numpy as np
import openmdao.api as om
from pyNA.src.engine import Engine
from openmdao.components.interp_util.interp import TABLE_METHODS


class TSLimit(om.MetaModelStructuredComp):
    """
    Computes aerodynamic lift and drag coefficients from aerodynamics data tables.

    The *TSLimit* component requires the following inputs:

    * ``inputs['z']``:                  aircraft angle of attack [deg]
    * ``inputs['v']``:                  aircraft flap deflection angle [deg]
    * ``inputs['theta_slats']``:        aircraft slat deflection angle [deg]

    The *TSLimit* component computes the following outputs:

    * ``outputs['TS_min']``:            aircraft lift coefficient [-]
    
    """

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('extrapolate', types=bool, default=False, desc='Sets whether extrapolation should be performed when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False, desc='Sets whether gradients with respect to output training data should be computed.')
        self.options.declare('vec_size', types=int, default=1, desc='Number of points to evaluate at once.')
        self.options.declare('method', values=TABLE_METHODS, default='scipy_cubic', desc='Spline interpolation method to use for all outputs.')
        self.options.declare('engine', types=Engine)

    def setup(self):
        # Load options
        nn = self.options['vec_size']
        engine = self.options['engine']

        self.add_input('z', val=np.ones(nn), units='m', training_data=engine.TS_limit['z'])
        self.add_input('v', val=np.ones(nn), units='m/s', training_data=engine.TS_limit['v'])
        self.add_input('theta_flaps', val=np.ones(nn), units='deg', training_data=engine.TS_limit['theta_flaps'])
        
        self.add_output('TS_min', val=np.ones(nn), units=None, training_data=engine.TS_limit['TS_min'])
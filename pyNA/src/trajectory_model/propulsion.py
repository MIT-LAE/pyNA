import pdb
import openmdao.api as om
import numpy as np
from pyNA.src.engine import Engine
from openmdao.components.interp_util.interp import TABLE_METHODS


class Propulsion(om.MetaModelStructuredComp):
    """
    Interpolates engine parameters from engine deck.

    The *Propulsion* component requires the following inputs:

    * ``inputs['z']``:                  aircraft z-position [m]
    * ``inputs['M_0']``:                ambient Mach number [-]
    * ``inputs['tau']``:                 engine thrust-setting [-]

    The *Propulsion* component computes the following outputs:

    * ``outputs['var']``:               all variables from the self.options['engine'].deck_variables
    
    """

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('vec_size', types=int, default=1, desc='Number of points to evaluate at once.')
        self.options.declare('extrapolate', types=bool, default=False, desc='Sets whether extrapolation should be performed when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False, desc='Sets whether gradients with respect to output training data should be computed.')
        self.options.declare('method', values=TABLE_METHODS, default='scipy_cubic', desc='Spline interpolation method to use for all outputs.')
        self.options.declare('engine', types=Engine)
        self.options.declare('atmosphere_mode', types=str)

    def setup(self):
        # Load options
        nn = self.options['vec_size']
        engine = self.options['engine']

        if self.options['atmosphere_mode'] == 'stratified':
            self.add_input('z', val=np.ones(nn), units='m', training_data=engine.deck['z'])
        self.add_input('M_0', val=np.ones(nn), units=None, training_data=engine.deck['M_0'])
        self.add_input('tau', val=np.ones(nn), units=None, training_data=engine.deck['TS'])

        for var in engine.var:
            self.add_output(var, val=np.ones(nn), units=engine.var_units[var], training_data=engine.deck[var])

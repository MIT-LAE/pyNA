import pdb
import numpy as np
import openmdao
import openmdao.api as om


class TrajectoryData(om.ExplicitComponent):
    """
    Create trajectory model from time history data.

    """

    def initialize(self):
        # Declare data option
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('vars', types=list, desc='trajectory output variables')
        self.options.declare('var_units', types=dict, desc='trajectory output variables units')

    def setup(self):

        # Load options
        nn = self.options['num_nodes']
        vars = self.options['vars']
        var_units = self.options['var_units']

        for var in vars:
            self.add_output(var, val=np.ones(nn,), units=var_units[var])

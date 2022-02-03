import pdb
import openmdao
import openmdao.api as om
import numpy as np


class SurrogateNoise(om.ExplicitComponent):
    """
    Compute minimum distance between the aircraft and the observer position.

    The *SurrogateNoise* component requires the following inputs:

    * ``inputs['x']``:                  aircraft x-position [m]
    * ``inputs['y']``:                  aircraft y-position [m]
    * ``inputs['z']``:                  aircraft z-position [m]

    The *SurrogateNoise* component computes the following outputs:

    * ``outputs['noise']``:         ambient pressure [Pa]

    """

    def initialize(self):
        # Declare data option
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('x_observer', types=np.ndarray, desc='Observer position [m,m,m]')

    def softmax(x, hardness=1):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(hardness*(x - np.max(x)))
        return e_x / e_x.sum()

    def setup(self):

        # Load options
        nn = self.options['num_nodes']

        # Add inputs and outputs
        self.add_input('x', val=np.ones(nn), units='m', desc='aircraft x-position [m]')
        self.add_input('y', val=np.ones(nn), units='m', desc='aircraft y-position [m]')
        self.add_input('z', val=np.ones(nn), units='m', desc='aircraft z-position [m]')
        self.add_input('t_s', val=np.ones(nn), units='s', desc='source time')

        self.add_output('noise', val=0., units=None, desc='Surrogate of noise due to propagation effects')

    def setup_partials(self):
        """
        Declare partials.
        """

        self.declare_partials('noise', 'x', method='cs')
        self.declare_partials('noise', 'y', method='cs')
        self.declare_partials('noise', 'z', method='cs')
        self.declare_partials('noise', 't_s', method='cs')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Extract inputs
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        t_s = inputs['t_s']
        
        # Compute distance to the observer
        r_obs = np.sqrt((x-self.options['x_observer'][0])**2 + (y-self.options['x_observer'][1])**2 + (z-self.options['x_observer'][2])**2)
        
        # Integrate the soft_r function over time
        msap = (6500/r_obs)**2
        outputs['noise'] = np.trapz(msap, t_s)
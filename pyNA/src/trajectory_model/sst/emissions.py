import pdb
import openmdao
import openmdao.api as om
import numpy as np


class Emissions(om.ExplicitComponent):
    """
    Compute emission characteristics of engine on the trajectory.

    The *Emissions* component requires the following inputs:

    * ``inputs['Tti_c']``:      combustor inlet temperature (re. T_0) [-]
    * ``inputs['Pti_c']``:      combustor inlet pressure (re. P_0) [-]
    * ``inputs['W_f']``:        engine fuel flow [kg/s]
    
    The *Emissions* component computes the following outputs:

    * ``outputs['EINOx']``:     NOx emission index [g/kg fuel]
    * ``outputs['m_NOx']``:     total mass of NOx emitted on the trajectory [kg]

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        # Load options
        nn = self.options['num_nodes']

        # Define inputs
        self.add_input('core_Tt_i', val=np.ones(nn), units='K', desc='combustor inlet temperature (re. T_0) [-]')
        self.add_input('core_Pt_i', val=np.ones(nn), units='Pa', desc='combustor inlet pressure (re. P_0) [-]')
        self.add_input('W_f', val=np.ones(nn), units='kg/s', desc='engine fuel flow [kg/s]')

        # Define outputs
        self.add_output('EINOx', val=np.ones(nn), units=None, desc='NOx emission index [g/kg fuel]')
        self.add_output('mdot_NOx', val=np.ones(nn), units='kg/s', desc='NOx mass flow emitted along the trajectory [kg/s]')

        # Define partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='EINOx', wrt='core_Pt_i', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='EINOx', wrt='core_Tt_i', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='mdot_NOx', wrt='core_Pt_i', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='mdot_NOx', wrt='core_Tt_i', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='mdot_NOx', wrt='W_f', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Extract inputs
        T3 = inputs['core_Tt_i']
        P3 = inputs['core_Pt_i']

        # Curve fit for EINOx for CFM56-5B3 engine from EDB (P3 in Pa and T3 in K)
        a = 6.25528852e-08
        b = -1.17064467e-04
        c = 7.36953400e-02
        d = -1.50392850e+01
        outputs['EINOx'] = (P3 / 1000.) ** 0.4 * (a * T3 ** 3 + b * T3 ** 2 + c * T3 + d)

        # Compute NOx emission mass flow
        outputs['mdot_NOx'] = outputs['EINOx'] * inputs['W_f']

    def compute_partials(self, inputs, partials):
        
        # Extract inputs
        T3 = inputs['core_Tt_i']
        P3 = inputs['core_Pt_i']

        # Curve fit for EINOx for CFM56-5B3 engine from EDB (P3 in Pa and T3 in K)
        a = 6.25528852e-08
        b = -1.17064467e-04
        c = 7.36953400e-02
        d = -1.50392850e+01
        EINOx = (P3 / 1000.) ** 0.4 * (a * T3 ** 3 + b * T3 ** 2 + c * T3 + d)

        partials['EINOx', 'core_Pt_i'] = 0.4 * (P3 / 1000.) ** (0.4-1) * (1/1000) * (a * T3 ** 3 + b * T3 ** 2 + c * T3 + d)
        partials['EINOx', 'core_Tt_i'] = (P3 / 1000.) ** 0.4 * (3* a * T3 ** 2 + 2* b * T3 + c)
        partials['mdot_NOx', 'core_Pt_i'] = inputs['W_f'] * 0.4 * (P3 / 1000.) ** (0.4-1) * (1/1000) * (a * T3 ** 3 + b * T3 ** 2 + c * T3 + d)
        partials['mdot_NOx', 'core_Tt_i'] = inputs['W_f'] * (P3 / 1000.) ** 0.4 * (3* a * T3 ** 2 + 2* b * T3 + c)
        partials['mdot_NOx', 'W_f'] = EINOx
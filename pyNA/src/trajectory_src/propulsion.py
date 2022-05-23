import pdb
import openmdao.api as om
import numpy as np
from pyNA.src.engine import Engine
from pyNA.src.settings import Settings
from openmdao.components.interp_util.interp import TABLE_METHODS


class Propulsion(om.MetaModelStructuredComp):
    """
    Interpolates engine parameters from engine deck.

    The *Propulsion* component requires the following inputs:

    * ``inputs['z']``:              aircraft z-position [m]
    * ``inputs['M_0']``:            ambient Mach number [-]
    * ``inputs['TS']``:             engine thrust-setting [-]

    The *Propulsion* component computes the following outputs:

    * ``outputs['W_f']``:           engine fuel consumption
    * ``outputs['F_n']``:           engine net thrust [N]
    * ``outputs['Tti_c']``:         combustor inlet total temperature [K]
    * ``outputs['Pti_c']``:         combustor inlet total pressure [Pa]
    * ``outputs['V_j']``:           engine jet velocity [m/s]
    * ``outputs['rho_j']``:         engine jet density [kg/m3]
    * ``outputs['A_j']``:           engine jet area [m2]
    * ``outputs['Tt_j']``:          engine jet total temperature [K]
    * ``outputs['M_j']``:           engine jet Mach number [-]
    * ``outputs['mdoti_c']``:       engine combustor inlet mass flow [kg/s]
    * ``outputs['Ttj_c']``:         engine combustor exit total temperature [K]
    * ``outputs['DTt_des_c']``:     engine design turbine total temperature drop [K]
    * ``outputs['rho_ti_c']``:      engine turbine inlet density [kg/m3]
    * ``outputs['c_ti_c']``:        engine turbine inlet speed of sound [m/s]
    * ``outputs['rho_te_c']``:      engine turbine exit density [kg/m3]
    * ``outputs['c_te_c']``:        engine turbine exit speed of sound [m/s]
    * ``outputs['DTt_f']``:         engine fan total temperature rise [K]
    * ``outputs['mdot_f']``:        engine fan mass flow [kg/s]
    * ``outputs['N_f']``:           engien fan rotational speed [rpm]
    * ``outputs['A_f']``:           engien fan inlet area [m2]
    * ``outputs['d_f']``:           engien fan diameter [m]

    """

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('extrapolate', types=bool, default=False, desc='Sets whether extrapolation should be performed when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False, desc='Sets whether gradients with respect to output training data should be computed.')
        self.options.declare('vec_size', types=int, default=1, desc='Number of points to evaluate at once.')
        self.options.declare('method', values=TABLE_METHODS, default='scipy_cubic', desc='Spline interpolation method to use for all outputs.')
        self.options.declare('settings', types=Settings)
        self.options.declare('engine', types=Engine)

    def setup(self):
        # Load options
        nn = self.options['vec_size']
        engine = self.options['engine']
        settings = self.options['settings']

        self.add_input('z', val=np.ones(nn), units='m', training_data=engine.deck['z'])
        self.add_input('M_0', val=np.ones(nn), units=None, training_data=engine.deck['M_0'])
        self.add_input('TS', val=np.ones(nn), units=None, training_data=engine.deck['TS'])

        self.add_output('W_f', val=np.ones(nn), units='kg/s', training_data=engine.deck['W_f'])
        self.add_output('F_n', val=np.ones(nn),units='N', training_data=engine.deck['F_n'])
        self.add_output('Tti_c', val=np.ones(nn), units='degK', training_data=engine.deck['Tti_c'], desc='core inlet total temperature [K]')
        self.add_output('Pti_c', val=np.ones(nn), units='Pa', training_data=engine.deck['Pti_c'], desc='core inlet total pressure [Pa]')

        # Jet parameters
        self.add_output('V_j', val=np.ones(nn), units='m/s', training_data=engine.deck['V_j'], desc='jet velocity (re. c_0) [-]')
        self.add_output('rho_j', val=np.ones(nn), units='kg/m**3', training_data=engine.deck['rho_j'], desc='jet total density (re. rho_0) [-]')
        self.add_output('A_j', val=np.ones(nn), units='m**2', training_data=engine.deck['A_j'], desc='jet area (re. A_e) [-]')
        self.add_output('Tt_j', val=np.ones(nn), units='degK', training_data=engine.deck['Tt_j'], desc='jet total temperature (re. T_0) [-]')
        self.add_output('M_j', val=np.ones(nn), units=None, training_data=engine.deck['M_j'], desc='jet mach number [-]')
        
        # Core parameters
        if settings.method_core_turb == 'GE':
            self.add_output('mdoti_c', val=np.ones(nn), units='kg/s', training_data=engine.deck['mdot_i_c'], desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
            self.add_output('Ttj_c', val=np.ones(nn), units='degK', training_data=engine.deck['Ttj_c'], desc='core exit total temperature (re. T_0) [-]')
            self.add_output('DTt_des_c', val=np.ones(nn), units='degK', training_data=engine.deck['DTt_des_c'], desc='core total temperature drop across the turbine (re. T_0) [-]')
        elif settings.method_core_turb == 'PW':                    
            self.add_output('mdoti_c', val=np.ones(nn), units='kg/s', training_data=engine.deck['mdoti_c'], desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
            self.add_output('Ttj_c', val=np.ones(nn), units='degK', training_data=engine.deck['Ttj_c'], desc='core exit total temperature (re. T_0) [-]')
            self.add_output('rho_te_c', val=np.ones(nn), units='kg/m**3', training_data=engine.deck['rho_te_c'], desc='core exit total density (re. rho_0) [-]')
            self.add_output('c_te_c', val=np.ones(nn), units='m/s', training_data=engine.deck['c_te_c'], desc='core exit total speed of sound (re. c_0) [-]')
            self.add_output('rho_ti_c', val=np.ones(nn), units='kg/m**3', training_data=engine.deck['rho_ti_c'], desc='core inlet total density (re. rho_0) [-]')
            self.add_output('c_ti_c', val=np.ones(nn), units='m/s', training_data=engine.deck['c_ti_c'], desc='core inlet total speed of sound (re. c_0) [-]')

        # Fan parameters
        self.add_output('DTt_f', val=np.ones(nn), units='degK', training_data=engine.deck['DTt_f'], desc='fan total temperature rise (re. T_0) [-]')
        self.add_output('mdot_f', val=np.ones(nn), units='kg/s', training_data=engine.deck['mdot_f'], desc='fan inlet mass flow (re. rho_0c_0A_e) [-]')
        self.add_output('N_f', val=np.ones(nn), units='rpm', training_data=engine.deck['N_f'], desc='fan rotational speed (re. c_0/sqrt(A_e)) [-]')
        self.add_output('A_f', val=np.ones(nn), units='m**2', training_data=engine.deck['A_f'], desc='fan inlet area [m2]')
        self.add_output('d_f', val=np.ones(nn), units='m', training_data=engine.deck['d_f'], desc='fan diameter [m]')
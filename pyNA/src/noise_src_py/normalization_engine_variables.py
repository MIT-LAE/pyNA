import pdb
import openmdao
import pandas as pd
import openmdao.api as om
import numpy as np
from pyNA.src.settings import Settings

## CHECK THIS COMPONENT

class NormalizationEngineVariables(om.ExplicitComponent):
    """
    Normalizes engine variables with ambient atmospheric parameters.

    The *NormalizationEngineVariables* component requires the following inputs:

    * ``inputs['c_0']``:            ambient speed of sound [m/s]
    * ``inputs['T_0']``:            ambient temperature [K]
    * ``inputs['p_0']``:            ambient pressure [Pa]
    * ``inputs['rho_0']``:          ambient density [kg/m3]

    * ``inputs['F_n']``:            engine net thrust [N]
    * ``inputs['W_f']``:            engine fuel mass flow [kg/s]
    * ``inputs['V_j']``:            jet velocity (re. c_0) [-]
    * ``inputs['rho_j']``:          jet total density (re. rho_0) [-]
    * ``inputs['A_j']``:            jet area (re. A_e) [-]
    * ``inputs['Tt_j']``:           jet total temperature (re. T_0) [-]
    * ``inputs['M_j']``:            jet Mach number [-]
    * ``inputs['mdoti_c']``:        core inlet mass flow (re. rho_0c_0A_e) [-]
    * ``inputs['Tti_c']``:          core inlet total temperature (re. T_0) [-]
    * ``inputs['Ttj_c']``:          core exit total temperature (re. T_0) [-]
    * ``inputs['Pti_c']``:          core inlet total pressure (re. p_O) [-]
    * ``inputs['DTt_des_c']``:      core total temperature drop across the turbine (re. T_0) [-]
    * ``inputs['rho_te_c']``:       core exit total density (re. rho_0) [-]
    * ``inputs['c_te_c']``:         core exit total speed of sound (re. c_0) [-]
    * ``inputs['rho_ti_c']``:       core inlet total density (re. rho_0) [-]
    * ``inputs['c_te_c']``:         core inlet total speed of sound (re. c_0) [-]
    * ``inputs['mdot_f']``:         fan inlet mass flow (re. rho_0c_0A_e) [-]
    * ``inputs['N_f']``:            fan rotational speed (re. c_0/sqrt(A_e)) [-]
    * ``inputs['d_f']``:            fan diameter (re. sqrt(A_e)) [-]
    * ``inputs['DTt_f']``:          fan total temperature rise (re. T_0) [-]

    The *NormalizationEngineVariables* component computes the following outputs:

    * ``outputs['F_n']``:            engine net thrust [N]
    * ``outputs['W_f']``:            engine fuel mass flow [kg/s]
    * ``outputs['V_j_star']``:       jet velocity (re. c_0) [-]
    * ``outputs['rho_j_star']``:     jet total density (re. rho_0) [-]
    * ``outputs['A_j_star']``:       jet area (re. A_e) [-]
    * ``outputs['Tt_j_star']``:      jet total temperature (re. T_0) [-]
    * ``outputs['M_j']``:            jet Mach number [-]
    * ``outputs['mdoti_c_star']``:   core inlet mass flow (re. rho_0c_0A_e) [-]
    * ``outputs['Tti_c_star']``:     core inlet total temperature (re. T_0) [-]
    * ``outputs['Ttj_c_star']``:     core exit total temperature (re. T_0) [-]
    * ``outputs['Pti_c_star']``:     core inlet total pressure (re. p_O) [-]
    * ``outputs['DTt_des_c_star']``: core total temperature drop across the turbine (re. T_0) [-]
    * ``outputs['rho_te_c_star']``:  core exit total density (re. rho_0) [-]
    * ``outputs['c_te_c_star']``:    core exit total speed of sound (re. c_0) [-]
    * ``outputs['rho_ti_c_star']``:  core inlet total density (re. rho_0) [-]
    * ``outputs['c_te_c_star']``:    core inlet total speed of sound (re. c_0) [-]
    * ``outputs['mdot_f_star']``:    fan inlet mass flow (re. rho_0c_0A_e) [-]
    * ``outputs['N_f_star']``:       fan rotational speed (re. c_0/sqrt(A_e)) [-]
    * ``outputs['d_f_star']``:       fan diameter (re. sqrt(A_e)) [-]
    * ``outputs['A_f_star']``:       fan inlet area (re. A_e) [-]
    * ``outputs['DTt_f_star']``:     fan total temperature rise (re. T_0) [-]

    The *NormalizationEngineVariables* component has the following options: 

    * ``settings``:                 pyna settings
    * ``n_t``:                      number of time steps in the noise time series 

    """

    def initialize(self):
        self.options.declare('settings', types=Settings)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')

    def setup(self):
        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Add inputs
        self.add_input('c_0', val=np.ones(n_t), units='m/s', desc='ambient speed of sound [m/s]')
        self.add_input('T_0', val=np.ones(n_t), units='K', desc='ambient temperature [K]')
        self.add_input('p_0', val=np.ones(n_t), units='Pa', desc='ambient pressure [Pa]')
        self.add_input('rho_0', val=np.ones(n_t), units='kg/m**3', desc='ambient density [kg/m3]')

        if settings.jet_mixing and settings.jet_shock == False:
            self.add_input('V_j', val=np.ones(n_t), units='m/s', desc='jet velocity [m/s]')
            self.add_input('rho_j', val=np.ones(n_t), units='kg/m**3', desc='jet total density [kg/m3]')
            self.add_input('A_j', val=np.ones(n_t), units='m**2', desc='jet area [m2]')
            self.add_input('Tt_j', val=np.ones(n_t), units='K', desc='jet total temperature [K]')
        elif settings.jet_shock and settings.jet_mixing == False:
            self.add_input('V_j', val=np.ones(n_t), units='m/s', desc='jet velocity [m/s]')
            self.add_input('A_j', val=np.ones(n_t), units='m**2', desc='jet area [m2]')
            self.add_input('Tt_j', val=np.ones(n_t), units='K', desc='jet total temperature [K]')
        elif settings.jet_shock and settings.jet_mixing:
            self.add_input('V_j', val=np.ones(n_t), units='m/s', desc='jet velocity [m/s]')
            self.add_input('rho_j', val=np.ones(n_t), units='kg/m**3', desc='jet total density [kg/m3]')
            self.add_input('A_j', val=np.ones(n_t), units='m**2', desc='jet area [m2]')
            self.add_input('Tt_j', val=np.ones(n_t), units='K', desc='jet total temperature [K]')
        if settings.core:
            if settings.method_core_turb == 'GE':
                self.add_input('mdoti_c', val=np.ones(n_t), units='kg/s', desc='core inlet mass flow [kg/s]')
                self.add_input('Tti_c', val=np.ones(n_t), units='K', desc='core inlet total temperature [K]')
                self.add_input('Ttj_c', val=np.ones(n_t), units='K', desc='core exit total temperature [K]')
                self.add_input('Pti_c', val=np.ones(n_t), units='Pa', desc='core inlet total pressure [Pa]')
                self.add_input('DTt_des_c', val=np.ones(n_t), units='K', desc='core total temperature drop across the turbine [K]')
            elif settings.method_core_turb == 'PW':
                self.add_input('mdoti_c', val=np.ones(n_t), units='kg/m**3', desc='core inlet mass flow [kg/s]')
                self.add_input('Tti_c', val=np.ones(n_t), units='K', desc='core inlet total temperature [K]')
                self.add_input('Ttj_c', val=np.ones(n_t), units='K', desc='core exit total temperature [K]')
                self.add_input('Pti_c', val=np.ones(n_t), units='Pa', desc='core inlet total pressure [Pa]')
                self.add_input('rho_te_c', val=np.ones(n_t), units='kg/m**3', desc='core exit total density [kg/m3]')
                self.add_input('c_te_c', val=np.ones(n_t), units='m/s', desc='core exit total speed of sound [m/s]')
                self.add_input('rho_ti_c', val=np.ones(n_t), units='kg/m**3', desc='core inlet total density [kg/m3]')
                self.add_input('c_ti_c', val=np.ones(n_t), units='m/s', desc='core inlet total speed of sound [m/s]')
        if settings.fan_inlet or settings.fan_discharge:
            self.add_input('mdot_f', val=np.ones(n_t), units='kg/s', desc='fan inlet mass flow [kg/s]')
            self.add_input('N_f', val=np.ones(n_t), units='rpm', desc='fan rotational speed [rpm]')
            self.add_input('DTt_f', val=np.ones(n_t), units='K', desc='fan total temperature rise [K]')
            self.add_input('A_f', val=np.ones(n_t), units='m**2', desc='fan inlet area [m2]')
            self.add_input('d_f', val=np.ones(n_t), units='m', desc='fan diameter [m]')

        # Add outputs
        if settings.jet_mixing and settings.jet_shock == False:
            self.add_output('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_output('rho_j_star', val=np.ones(n_t), units=None, desc='jet total density (re. rho_0) [-]')
            self.add_output('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_output('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
        elif settings.jet_shock and settings.jet_mixing == False:
            self.add_output('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_output('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_output('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
            self.add_output('M_j', val=np.ones(n_t), units=None)
        elif settings.jet_shock and settings.jet_mixing:
            self.add_output('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_output('rho_j_star', val=np.ones(n_t), units=None, desc='jet total density (re. rho_0) [-]')
            self.add_output('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_output('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
        if settings.core:
            if settings.method_core_turb == 'GE':
                self.add_output('mdoti_c_star', val=np.ones(n_t), units=None, desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
                self.add_output('Tti_c_star', val=np.ones(n_t), units=None, desc='core inlet total temperature (re. T_0) [-]')
                self.add_output('Ttj_c_star', val=np.ones(n_t), units=None, desc='core exit total temperature (re. T_0) [-]')
                self.add_output('Pti_c_star', val=np.ones(n_t), units=None, desc='core inlet total pressure (re. p_O) [-]')
                self.add_output('DTt_des_c_star', val=np.ones(n_t), units=None, desc='core total temperature drop across the turbine (re. T_0) [-]')
            elif settings.method_core_turb == 'PW':
                self.add_output('mdoti_c_star', val=np.ones(n_t), units=None, desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
                self.add_output('Tti_c_star', val=np.ones(n_t), units=None, desc='core inlet total temperature (re. T_0) [-]')
                self.add_output('Ttj_c_star', val=np.ones(n_t), units=None, desc='core exit total temperature (re. T_0) [-]')
                self.add_output('Pti_c_star', val=np.ones(n_t), units=None, desc='core inlet total pressure (re. p_O) [-]')
                self.add_output('rho_te_c_star', val=np.ones(n_t), units=None, desc='core exit total density (re. rho_0) [-]')
                self.add_output('c_te_c_star', val=np.ones(n_t), units=None, desc='core exit total speed of sound (re. c_0) [-]')
                self.add_output('rho_ti_c_star', val=np.ones(n_t), units=None, desc='core inlet total density (re. rho_0) [-]')
                self.add_output('c_ti_c_star', val=np.ones(n_t), units=None, desc='core inlet total speed of sound (re. c_0) [-]')
        if settings.fan_inlet or settings.fan_discharge:
            self.add_output('mdot_f_star', val=np.ones(n_t), units=None, desc='fan inlet mass flow (re. rho_0c_0A_e) [-]')
            self.add_output('N_f_star', val=np.ones(n_t), units=None, desc='fan rotational speed (re. c_0/sqrt(A_e)) [-]')
            self.add_output('DTt_f_star', val=np.ones(n_t), units=None, desc='fan total temperature rise (re. T_0) [-]')
            self.add_output('A_f_star', val=np.ones(n_t), units=None, desc='fan inlet area (re. A_e) [-]')
            self.add_output('d_f_star', val=np.ones(n_t), units=None, desc='fan diameter (re. sqrt(A_e)) [-]')

        self.add_output('F_n', val=np.ones(n_t), units='N')
        self.add_output('W_f', val=np.ones(n_t), units='kg/s')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        settings = self.options['settings']

        if settings.jet_mixing and settings.jet_shock == False:
            outputs['V_j_star'] = inputs['V_j'] / inputs['c_0']
            outputs['rho_j_star'] = inputs['rho_j'] / inputs['rho_0']
            outputs['A_j_star'] = inputs['A_j'] / settings.A_e
            outputs['Tt_j_star'] = inputs['Tt_j'] / inputs['T_0']
        elif settings.jet_shock and settings.jet_mixing == False:
            outputs['V_j_star'] = inputs['V_j'] / inputs['c_0']
            outputs['A_j_star'] = inputs['A_j'] / settings.A_e
            outputs['Tt_j_star'] = inputs['Tt_j'] / inputs['T_0']
        elif settings.jet_shock and settings.jet_mixing:
            outputs['V_j_star'] = inputs['V_j'] / inputs['c_0']
            outputs['rho_j_star'] = inputs['rho_j'] / inputs['rho_0']
            outputs['A_j_star'] = inputs['A_j'] / settings.A_e
            outputs['Tt_j_star'] = inputs['Tt_j'] / inputs['T_0']
        
        if settings.core:
            if settings.method_core_turb == 'GE':
                outputs['mdoti_c_star'] = inputs['mdoti_c'] / (inputs['rho_0'] * inputs['c_0'] * settings.A_e)
                outputs['Tti_c_star'] = inputs['Tti_c'] / inputs['T_0']
                outputs['Ttj_c_star'] = inputs['Ttj_c'] / inputs['T_0']
                outputs['Pti_c_star'] = inputs['Pti_c'] / inputs['p_0']
                outputs['DTt_des_c_star'] = inputs['DTt_des_c'] / inputs['T_0']
            elif settings.method_core_turb == 'PW':    
                outputs['mdoti_c_star'] = inputs['mdoti_c'] / (inputs['rho_0'] * inputs['c_0'] * settings.A_e)
                outputs['Tti_c_star'] = inputs['Tti_c'] / inputs['T_0']
                outputs['Ttj_c_star'] = inputs['Ttj_c'] / inputs['T_0']
                outputs['Pti_c_star'] = inputs['Pti_c'] / inputs['p_0']
                outputs['rho_te_c_star'] = inputs['rho_te_c'] / inputs['rho_0']
                outputs['c_te_c_star'] = inputs['c_te_c'] / inputs['c_0']
                outputs['rho_ti_c_star'] = inputs['rho_ti_c'] / inputs['rho_0']
                outputs['c_ti_c_star'] = inputs['c_ti_c'] / inputs['c_0']

        if settings.fan_inlet or settings.fan_discharge:
            outputs['DTt_f_star'] = inputs['DTt_f'] / inputs['T_0']
            outputs['mdot_f_star'] = inputs['mdot_f'] / (inputs['rho_0'] * inputs['c_0'] * settings.A_e)
            outputs['d_f_star'] = inputs['d_f'] / np.sqrt(settings.A_e)
            outputs['A_f_star'] = inputs['A_f'] / settings.A_e
            outputs['N_f_star'] = inputs['N_f'] / (inputs['c_0'] / inputs['d_f'] * 60)

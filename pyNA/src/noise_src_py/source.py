import pdb
import openmdao
import numpy as np
import openmdao.api as om
from typing import Dict, Any
from pyNA.src.data import Data
from pyNA.src.aircraft import Aircraft
from pyNA.src.settings import Settings
from pyNA.src.noise_src_py.jet import Jet
from pyNA.src.noise_src_py.fan import Fan
from pyNA.src.noise_src_py.core import Core
from pyNA.src.noise_src_py.airframe import Airframe


class Source(om.ExplicitComponent):
    """
    Compute noise source mean-square acoustic pressure (msap):

    * Fan noise (inlet, discharge; tone and broadband)
    * Core noise
    * Jet noise (jet mixing and jet shock)
    * Airframe noise

    The *Source* component requires the following inputs:

    * ``inputs['V_j_star']``:               jet velocity (re. c_0) [-]
    * ``inputs['rho_j_star']``:             jet total density (re. rho_0) [-]
    * ``inputs['A_j_star']``:               jet area (re. A_e) [-]
    * ``inputs['Tt_j_star']``:              jet total temperature (re. T_0) [-]
    * ``inputs['M_j']``:                    jet Mach number [-]
    * ``inputs['mdoti_c_star']``:           core inlet mass flow (re. rho_0c_0A_e) [-]
    * ``inputs['Tti_c_star']``:             core inlet total temperature (re. T_0) [-]
    * ``inputs['Ttj_c_star']``:             core exit total temperature (re. T_0) [-]
    * ``inputs['Pti_c_star']``:             core inlet total pressure (re. p_O) [-]
    * ``inputs['DTt_des_c_star']``:         core total temperature drop across the turbine (re. T_0) [-]
    * ``inputs['rho_te_c']``:               core exit total density (re. rho_0) [-]
    * ``inputs['c_te_c']``:                 core exit total speed of sound (re. c_0) [-]
    * ``inputs['rho_ti_c']``:               core inlet total density (re. rho_0) [-]
    * ``inputs['c_te_c']``:                 core inlet total speed of sound (re. c_0) [-]
    * ``inputs['DTt_f_star']``:             fan total temperature rise (re. T_0) [-]
    * ``inputs['mdot_f_star']``:            fan inlet mass flow (re. rho_0c_0A_e) [-]
    * ``inputs['N_f_star']``:               fan rotational speed (re. c_0/sqrt(A_e)) [-]
    * ``inputs['A_f_star']``:               fan area (re. A_e) [-]
    * ``inputs['d_f_star']``                fan diameter (re. sqrt(A_e)) [-]
    * ``inputs['I_landing_gear']``:         airframe landing gear extraction (0/1) [-]
    * ``inputs['theta_flaps']``:            airframe flap angle [deg]
    * ``inputs['TS']``:                     engine power-setting [-]
    * ``inputs['M_0']``:                    aircraft Mach number [-]
    * ``inputs['c_0']``:                    ambient speed of sound [m/s]
    * ``inputs['rho_0']``:                  ambient density [kg/m3]
    * ``inputs['mu_0']``:                   ambient dynamic viscosity [kg/ms]
    * ``inputs['T_0']``:                    ambient temperature [K]
    * ``inputs['theta']``:                  polar directivity angle [deg]
    * ``inputs['phi']``:                    azimuthal directivity angle [deg]

    The *Source* component computes the following outputs:

    * ``inputs['msap_jet_mixing']``:        mean-square acoustic pressure of the jet mixing noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_jet_shock']``:         mean-square acoustic pressure of the jet shock noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_core']``:              mean-square acoustic pressure of the core noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_fan_inlet']``:         mean-square acoustic pressure of the fan inlet noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_fan_discharge']``:     mean-square acoustic pressure of the fan discharge noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_airframe']``:          mean-square acoustic pressure of the airframe noise source (re. rho_0^2 c_0^2) [-]
    * ``inputs['msap_source']``:            mean-square acoustic pressure of the overall noise source (re. rho_0^2 c_0^2) [-]

    The *Source* component has the followign options

    * ``settings``:                         pyna settings
    * ``n_t``:                              number of time steps in the noise time series 
    * ``data``:                             pyna noise data
    * ``ac``:                               aircraft characteristics

    """
    def initialize(self):
        # Declare data option
        self.options.declare('settings', types=Settings)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('data', types=Data)
        self.options.declare('ac', types=Aircraft)

    def setup(self):

        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings.x_observer_array)[0]

        # Inputs
        self.add_input('TS', val=np.ones(n_t), units=None, desc='engine power-setting [-]')
        self.add_input('M_0', val=np.ones(n_t), units=None, desc='aircraft Mach number [-]')
        self.add_input('c_0', val=np.ones(n_t), units='m/s', desc='ambient speed of sound [m/s]')
        self.add_input('rho_0', val=np.ones(n_t), units='kg/m**3', desc='ambient density [kg/m3]')
        self.add_input('mu_0', val=np.ones(n_t), units='kg/m/s', desc='ambient dynamic viscosity [kg/ms]')
        self.add_input('T_0', val=np.ones(n_t), units='K', desc='ambient temperature [K]')
        self.add_input('theta', val=np.ones((n_obs, n_t)), units='deg', desc='polar directivity angle [deg]')
        self.add_input('phi', val=np.ones((n_obs, n_t)), units='deg', desc='azimuthal directivity angle [deg]')
        self.add_input('shield', val=np.ones((n_obs, n_t, settings.N_f)), units=None, desc='shielding factors for the trajectory')

        if settings.jet_mixing and settings.jet_shock == False:
            self.add_input('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_input('rho_j_star', val=np.ones(n_t), units=None, desc='jet total density (re. rho_0) [-]')
            self.add_input('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_input('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
        elif settings.jet_shock and settings.jet_mixing == False:
            self.add_input('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_input('M_j', val=np.ones(n_t), units=None, desc='jet Mach number [-]')
            self.add_input('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_input('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
        elif settings.jet_shock and settings.jet_mixing:
            self.add_input('V_j_star', val=np.ones(n_t), units=None, desc='jet velocity (re. c_0) [-]')
            self.add_input('rho_j_star', val=np.ones(n_t), units=None, desc='jet total density (re. rho_0) [-]')
            self.add_input('A_j_star', val=np.ones(n_t), units=None, desc='jet area (re. A_e) [-]')
            self.add_input('Tt_j_star', val=np.ones(n_t), units=None, desc='jet total temperature (re. T_0) [-]')
            self.add_input('M_j', val=np.ones(n_t), units=None, desc='jet Mach number [-]')
        if settings.core:
            if settings.method_core_turb == 'GE':
                self.add_input('mdoti_c_star', val=np.ones(n_t), units=None, desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
                self.add_input('Tti_c_star', val=np.ones(n_t), units=None, desc='core inlet total temperature (re. T_0) [-]')
                self.add_input('Ttj_c_star', val=np.ones(n_t), units=None, desc='core exit total temperature (re. T_0) [-]')
                self.add_input('Pti_c_star', val=np.ones(n_t), units=None, desc='core inlet total pressure (re. p_0) [-]')
                self.add_input('DTt_des_c_star', val=np.ones(n_t), units=None, desc='core total temperature drop across the turbine (re. T_0) [-]')
            elif settings.method_core_turb == 'PW':
                self.add_input('mdoti_c_star', val=np.ones(n_t), units=None, desc='core inlet mass flow (re. rho_0c_0A_e) [-]')
                self.add_input('Tti_c_star', val=np.ones(n_t), units=None, desc='core inlet total temperature (re. T_0) [-]')
                self.add_input('Ttj_c_star', val=np.ones(n_t), units=None, desc='core exit total temperature (re. T_0) [-]')
                self.add_input('Pti_c_star', val=np.ones(n_t), units=None, desc='core inlet total pressure (re. p_0) [-]')
                self.add_input('rho_te_c_star', val=np.ones(n_t), units=None, desc='core exit total density (re. rho_0) [-]')
                self.add_input('c_te_c_star', val=np.ones(n_t), units=None, desc='core exit total speed of sound (re. c_0) [-]')
                self.add_input('rho_ti_c_star', val=np.ones(n_t), units=None, desc='core inlet total density (re. rho_0) [-]')
                self.add_input('c_ti_c_star', val=np.ones(n_t), units=None, desc='core inlet total speed of sound (re. c_0) [-]')
        if settings.airframe:
            self.add_input('theta_flaps', val=np.ones(n_t), units='deg', desc='airframe flap angle [deg]')
            self.add_input('I_landing_gear', val=np.ones(n_t), units=None, desc='airframe landing gear extraction (0/1) [-]')
        if settings.fan_inlet or settings.fan_discharge:
            self.add_input('DTt_f_star', val=np.ones(n_t), units=None, desc='fan total temperature rise (re. T_0) [-]')
            self.add_input('mdot_f_star', val=np.ones(n_t), units=None, desc='fan inlet mass flow (re. rho_0c_0A_e) [-]')
            self.add_input('N_f_star', val=np.ones(n_t), units=None, desc='fan rotational speed (re. c_0/sqrt(A_e)) [-]')
            self.add_input('A_f_star', val=np.ones(n_t), units=None, desc='fan area (re. A_e) [-]')
            self.add_input('d_f_star', val=np.ones(n_t), units=None, desc='fan diameter (re. sqrt(A_e) [-]')

        # Output
        self.add_output('msap_source', val=np.zeros((n_obs, n_t, settings.N_f)), desc='mean-square acoustic pressure of the overall noise source (re. rho_0,^2c_0^2) [-]')

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']

        # Number of observers
        n_obs = np.shape(settings.x_observer_array)[0]

        # Initialize sourde msap
        msap_source = np.zeros((n_obs, n_t, settings.N_f))

        for i in np.arange(n_obs):

            if settings.fan_inlet:
                msap_fan_inlet = Fan.fan(self, inputs['theta'][i, :], inputs['shield'][i,:,:], inputs, 'fan_inlet')
                msap_source[i, :, :] = msap_source[i, :, :] + msap_fan_inlet

            if settings.fan_discharge:
                msap_fan_discharge = Fan.fan(self, inputs['theta'][i, :], inputs['shield'][i,:,:], inputs, 'fan_discharge')
                msap_source[i, :, :] = msap_source[i, :, :] + msap_fan_discharge

            if settings.core:
                msap_core = Core.core(self, inputs['theta'][i, :], inputs)
                
                if settings.suppression and settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]:
                    idx_TS = np.where(np.reshape(inputs['TS'], (n_t, 1))*np.ones((1, settings.N_f)) > 0.8)
                    msap_core[idx_TS] = (10.**(-2.3 / 10.) * msap_core)[idx_TS]
                msap_source[i, :, :] = msap_source[i, :, :] + msap_core

            if settings.jet_mixing:
                msap_jet_mixing = Jet.jet_mixing(self, inputs['theta'][i, :], inputs)
                
                if settings.suppression and settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]:
                    idx_TS = np.where(np.reshape(inputs['TS'], (n_t, 1))*np.ones((1, settings.N_f)) > 0.8)
                    msap_jet_mixing[idx_TS] = (10. **(-2.3 / 10.) * msap_jet_mixing)[idx_TS]
                msap_source[i, :, :] = msap_source[i, :, :] + msap_jet_mixing

            if settings.jet_shock:
                msap_jet_shock = Jet.jet_shock(self, inputs['theta'][i, :], inputs)
                
                if settings.suppression and settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]:
                    idx_TS = np.where(np.reshape(inputs['TS'], (n_t, 1))*np.ones((1, settings.N_f)) > 0.8)
                    msap_jet_shock[idx_TS] = (10. **(-2.3 / 10.) * msap_jet_shock)[idx_TS]
                msap_source[i, :, :] = msap_source[i, :, :] + msap_jet_shock
            
            if settings.airframe:
                msap_airframe = Airframe.airframe(self, inputs['theta'][i, :], inputs['phi'][i, :], inputs)
                msap_source[i, :, :] = msap_source[i, :, :] + msap_airframe

        # Convert to dB (clip values to avoid -inf)
        outputs['msap_source'] = msap_source.clip(min=1e-99)
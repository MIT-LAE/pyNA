import pdb
import os
import copy
import pandas as pd
import dymos as dm
import numpy as np
import openmdao.api as om
from pyNA.src.noise_src_jl.get_noise_input_vector_indices import get_input_vector_indices
import matplotlib.pyplot as plt
from pyNA.src.data import Data
from pyNA.src.airframe import Airframe
from pyNA.src.engine import Engine

if os.environ['pyna_language']=='julia':
    import julia.Main as julia
    from julia.OpenMDAO import make_component
    src_path = os.path.dirname(os.path.abspath(__file__))
    julia.include(src_path + "/noise_src_jl/noise_model.jl")
elif os.environ['pyna_language']=='python':
    from pyNA.src.noise_src_py.noise_model import NoiseModel

class NoiseTimeSeries(om.Problem):

    def __init__(self, pyna_directory, case_name, language, save_results, settings, data, model=None, driver=None, comm=None, name=None, **options):
        super().__init__(model, driver, comm, name, **options)

        self.pyna_directory = pyna_directory
        self.case_name = case_name
        self.language = language
        self.save_results = save_results

        self.settings = settings
        self.data = data

    def create(self, airframe:Airframe, n_t:np.int, mode:str, objective:str) -> None:
        """
        Create model for computing noise of predefined trajectory timeseries.

        :param airframe: aircraft parameters
        :type airframe: Airframe
        :param n_t: Number of time steps in trajectory
        :type n_t: Dict[str, Any]
        :param mode: 
        :type mode: str
        :param objective: optimization objective
        :type objective: str

        :return: None
        """

        # Set solver
        self.model.linear_solver = om.LinearRunOnce()

        if self.language == 'python':

            self.model.add_subsystem(name='noise',
                                        subsys=NoiseModel(settings=self.settings, data=self.data, airframe=airframe, n_t=n_t, mode=mode), 
                                        promotes_inputs=[],
                                        promotes_outputs=[])
        
        elif self.language == 'julia':
            idx = get_input_vector_indices(self.language, settings=self.settings, n_t=n_t)
            
            self.model.add_subsystem(name='noise',
                                        subsys=make_component(julia.NoiseModel(self.settings, self.data, airframe, n_t, idx, objective)),
                                        promotes_inputs=[],
                                        promotes_outputs=[])

        return

    def solve(self, path: pd.DataFrame, engine: Engine, mode: str) -> None:
        """
        Compute noise of predefined trajectory time series.

        :param path: path of trajectory time series
        :type path: pd.DataFrame
        :param engine: 
        :type engine: Engine
        :param mode: 
        :type mode: str

        :return: None
        """

        # Run the openMDAO problem setup
        self.setup()

        # Attach a recorder to the problem to save model data
        if self.save_results:
            self.add_recorder(om.SqliteRecorder(self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.settings['output_file_name']))

        if self.language == 'python':
            # Set variables for engine normalization
            self['noise.normalize_engine.c_0'] = path['c_0 [m/s]']
            self['noise.normalize_engine.T_0'] = path['T_0 [K]']
            self['noise.normalize_engine.p_0'] = path['p_0 [Pa]']
            self['noise.normalize_engine.rho_0'] = path['rho_0 [kg/m3]']
            if self.settings['jet_mixing_source'] and not self.settings['jet_shock_source']:
                self['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                self['noise.normalize_engine.rho_j'] = engine['Jet rho [kg/m3]']
                self['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                self['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            elif self.settings['jet_shock_source'] and not self.settings['jet_mixing_source']:
                self['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                self['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                self['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            elif self.settings['jet_shock_source'] and self.settings['jet_mixing_source']:
                self['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                self['noise.normalize_engine.rho_j'] = engine['Jet rho [kg/m3]']
                self['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                self['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            if self.settings['core_source']:
                if self.settings['core_turbine_attenuation_method'] == "ge":
                    self['noise.normalize_engine.mdoti_c'] = engine['Core mdot [kg/s]']
                    self['noise.normalize_engine.Tti_c'] = engine['Core Tti [K]']
                    self['noise.normalize_engine.Ttj_c'] = engine['Core Ttj [K]']
                    self['noise.normalize_engine.Pti_c'] = engine['Core Pt [Pa]']
                    self['noise.normalize_engine.DTt_des_c'] = engine['Core DT_t [K]']
                elif self.settings['core_turbine_attenuation_method'] == "pw":
                    self['noise.normalize_engine.mdoti_c'] = engine['Core mdot [kg/s]']
                    self['noise.normalize_engine.Tti_c'] = engine['Core Tti [K]']
                    self['noise.normalize_engine.Ttj_c'] = engine['Core Ttj [K]']
                    self['noise.normalize_engine.Pti_c'] = engine['Core Pt [Pa]']
                    self['noise.normalize_engine.rho_te_c'] = engine['LPT rho_e [kg/m3]']
                    self['noise.normalize_engine.c_te_c'] = engine['LPT c_e [m/s]']
                    self['noise.normalize_engine.rho_ti_c'] = engine['HPT rho_i [kg/m3]']
                    self['noise.normalize_engine.c_ti_c'] = engine['HPT c_i [m/s]']
            if self.settings['fan_inlet_source'] or self.settings['fan_discharge_source']:
                self['noise.normalize_engine.DTt_f'] = engine['Fan delta T [K]']
                self['noise.normalize_engine.mdot_f'] = engine['Fan mdot in [kg/s]']
                self['noise.normalize_engine.N_f'] = engine['Fan N [rpm]']
                self['noise.normalize_engine.A_f'] = engine['Fan A [m2]']
                self['noise.normalize_engine.d_f'] = engine['Fan d [m]']
            # Set variables for geometry
            self['noise.geometry.x'] = path['X [m]']
            self['noise.geometry.y'] = path['Y [m]']
            self['noise.geometry.z'] = path['Z [m]']
            self['noise.geometry.alpha'] = path['alpha [deg]']
            self['noise.geometry.gamma'] = path['gamma [deg]']
            self['noise.geometry.c_0'] = path['c_0 [m/s]']
            self['noise.geometry.T_0'] = path['T_0 [K]']
            self['noise.geometry.t_s'] = path['t_source [s]']
            # Set variables for source
            self['noise.source.TS'] = path['TS [-]']
            self['noise.source.M_0'] = path['M_0 [-]']
            self['noise.source.c_0'] = path['c_0 [m/s]']
            self['noise.source.rho_0'] = path['rho_0 [kg/m3]']
            self['noise.source.mu_0'] = path['mu_0 [kg/ms]']
            self['noise.source.T_0'] = path['T_0 [K]']
            if self.settings['jet_shock_source'] and not self.settings['jet_mixing_source']:
                self['noise.source.M_j'] = engine['Jet M [-]']
            elif self.settings['jet_shock_source'] and self.settings['jet_mixing_source']:
                self['noise.source.M_j'] = engine['Jet M [-]']
            if self.settings['airframe_source']:
                self['noise.source.theta_flaps'] = path['Airframe delta_f [deg]']
                self['noise.source.I_landing_gear'] = path['Airframe LG [-]']
            # Set variables for propagation
            if mode == 'trajectory':
                self['noise.propagation.x'] = path['X [m]']
                self['noise.propagation.z'] = path['Z [m]']
                self['noise.propagation.rho_0'] = path['rho_0 [kg/m3]']
                self['noise.propagation.I_0'] = path['I_0 [kg/m2s]']
            # Set variables for levels
            self['noise.levels.c_0'] = path['c_0 [m/s]']
            self['noise.levels.rho_0'] = path['rho_0 [kg/m3]']

        elif self.language == 'julia':
            # Set path parameters
            self['noise.x'] = path['X [m]']
            self['noise.y'] = path['Y [m]']
            self['noise.z'] = path['Z [m]']
            self['noise.alpha'] = path['alpha [deg]']
            self['noise.gamma'] = path['gamma [deg]']
            self['noise.t_s'] = path['t_source [s]']
            self['noise.rho_0'] = path['rho_0 [kg/m3]']
            self['noise.c_0'] = path['c_0 [m/s]']
            self['noise.mu_0'] = path['mu_0 [kg/ms]']
            self['noise.T_0'] = path['T_0 [K]']
            self['noise.p_0'] = path['p_0 [Pa]']
            self['noise.M_0'] = path['M_0 [-]']
            self['noise.I_0'] = path['I_0 [kg/m2s]']
            self['noise.TS'] = path['TS [-]']

            if self.settings['jet_mixing_source'] and not self.settings['jet_shock_source']:
                self['noise.V_j'] = engine['Jet V [m/s]']
                self['noise.rho_j'] = engine['Jet rho [kg/m3]']
                self['noise.A_j'] = engine['Jet A [m2]']
                self['noise.Tt_j'] = engine['Jet Tt [K]']
            elif self.settings['jet_shock_source'] and not self.settings['jet_mixing_source']:
                self['noise.V_j'] = engine['Jet V [m/s]']
                self['noise.M_j'] = engine['Jet M [-]']
                self['noise.A_j'] = engine['Jet A [m2]']
                self['noise.Tt_j'] = engine['Jet Tt [K]']
            elif self.settings['jet_shock_source'] and self.settings['jet_mixing_source']:
                self['noise.V_j'] = engine['Jet V [m/s]']
                self['noise.rho_j'] = engine['Jet rho [kg/m3]']
                self['noise.A_j'] = engine['Jet A [m2]']
                self['noise.Tt_j'] = engine['Jet Tt [K]']
                self['noise.M_j'] = engine['Jet M [-]']

            if self.settings['core_source']:
                if self.settings['core_turbine_attenuation_method'] == "ge":
                    self['noise.mdoti_c'] = engine['Core mdot [kg/s]']
                    self['noise.Tti_c'] = engine['Core Tti [K]']
                    self['noise.Ttj_c'] = engine['Core Ttj [K]']
                    self['noise.Pti_c'] = engine['Core Pt [Pa]']
                    self['noise.DTt_des_c'] = engine['Core DT_t [K]']
                elif self.settings['core_turbine_attenuation_method'] == "pw":
                    self['noise.mdoti_c'] = engine['Core mdot [kg/s]']
                    self['noise.Tti_c'] = engine['Core Tti [K]']
                    self['noise.Ttj_c'] = engine['Core Ttj [K]']
                    self['noise.Pti_c'] = engine['Core Pt [Pa]']
                    self['noise.rho_te_c'] = engine['LPT rho_e [kg/m3]']
                    self['noise.c_te_c'] = engine['LPT c_e [m/s]']
                    self['noise.rho_ti_c'] = engine['HPT rho_i [kg/m3]']
                    self['noise.c_ti_c'] = engine['HPT c_i [m/s]']

            if self.settings['airframe_source']:
                self['noise.theta_flaps'] = path['Airframe delta_f [deg]']
                self['noise.I_landing_gear'] = path['Airframe LG [-]']

            if self.settings['fan_inlet_source'] or self.settings['fan_discharge_source']:
                self['noise.DTt_f'] = engine['Fan delta T [K]']
                self['noise.mdot_f'] = engine['Fan mdot in [kg/s]']
                self['noise.N_f'] = engine['Fan N [rpm]']
                self['noise.A_f'] = engine['Fan A [m2]']
                self['noise.d_f'] = engine['Fan d [m]']

        # Run noise module
        dm.run_problem(self, run_driver=False, simulate=False, solution_record_file=self.pyna_directory + '/cases/' + self.case_name + '/dymos_solution.db')

        # Save the results
        if self.save_results:
            self.record(case_name='time_series')

        return

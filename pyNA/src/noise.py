import pdb
import pandas as pd
import dymos as dm
import numpy as np
import os
import openmdao.api as om
from typing import Dict, Any
from pyNA.src.data import Data
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.engine import Engine
if os.environ['pyna_language']=='julia':
    import julia.Main as julia
    from julia.OpenMDAO import make_component
    src_path = os.path.dirname(os.path.abspath(__file__))
    julia.include(src_path + "/noise_src_jl/noise_model.jl")
elif os.environ['pyna_language']=='python':
    from pyNA.src.noise_src_py.noise_model import NoiseModel

class Noise:

    """
    The noise module contains the methods to compute the noise signature used by pyNA.
    """

    def __init__(self, settings: Settings):
        """
        Initialize noise class

        :param settings: pyna settings
        :type settings: Settings

        """
        # Load pyna model data
        self.data = Data(settings=settings)

        # Initialize noise distribution
        self.source_distribution = dict()

        # Initialize noise contour
        self.contour = dict()

        # Initialize epnl_grid
        self.epnl_table = pd.DataFrame()

    @staticmethod
    def get_indices_noise_input_vector(settings, n_t):
        """
        Get (julia) indices for input vector noise model.
        """

        # Initialize indices dictionary
        idx = dict()
        idx_src = dict()

        if settings.language == 'julia':
            idx["x"] = [0 * n_t + 1, 1 * n_t]
            idx["y"] = [1 * n_t + 1, 2 * n_t]
            idx["z"] = [2 * n_t + 1, 3 * n_t]
            idx["alpha"] = [3 * n_t + 1, 4 * n_t]
            idx["gamma"] = [4 * n_t + 1, 5 * n_t]
            idx["t_s"] = [5 * n_t + 1, 6 * n_t]
            idx["rho_0"] = [6 * n_t + 1, 7 * n_t]
            idx["mu_0"] = [7 * n_t + 1, 8 * n_t]
            idx["c_0"] = [8 * n_t + 1, 9 * n_t]
            idx["T_0"] = [9 * n_t + 1, 10 * n_t]
            idx["p_0"] = [10 * n_t + 1, 11 * n_t]
            idx["M_0"] = [11 * n_t + 1, 12 * n_t]
            idx["I_0"] = [12 * n_t + 1, 13 * n_t]
            idx["TS"] = [13 * n_t + 1, 14 * n_t]

            idx_src["TS"] = [0 * n_t + 1, 1 * n_t]
            idx_src["M_0"] = [1 * n_t + 1, 2 * n_t]
            idx_src["c_0"] = [2 * n_t + 1, 3 * n_t]
            idx_src["rho_0"] = [3 * n_t + 1, 4 * n_t]
            idx_src["mu_0"] = [4 * n_t + 1, 5 * n_t]
            idx_src["T_0"] = [5 * n_t + 1, 6 * n_t]
            idx_src["theta"] = [6 * n_t + 1, 7 * n_t]
            idx_src["phi"] = [7 * n_t + 1, 8 * n_t]

            n = 14
            n_src = 8
            if settings.jet_mixing == True and settings.jet_shock == False:
                idx["V_j"]   = [n * n_t + 1, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"]   = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"]  = [(n + 3) * n_t + 1, (n + 4) * n_t]
                n = n + 4
                idx_src["V_j_star"]   = [n_src * n_t + 1, (n_src + 1) * n_t]
                idx_src["rho_j_star"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                idx_src["A_j_star"]   = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                idx_src["Tt_j_star"]  = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                n_src = n_src + 4
            elif settings.jet_shock == True and settings.jet_mixing == False:
                idx["V_j"]  = [n * n_t + 1, (n + 1) * n_t]
                idx["M_j"]       = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"]  = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                n = n + 4
                idx_src["V_j_star"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                idx_src["M_j"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                idx_src["A_j_star"] = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                idx_src["Tt_j_star"] = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                n_src = n_src + 4
            elif settings.jet_shock == True and settings.jet_mixing == True:
                idx["V_j"] = [n * n_t + 1, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                idx["M_j"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                n = n + 5
                idx_src["V_j_star"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                idx_src["rho_j_star"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                idx_src["A_j_star"] = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                idx_src["Tt_j_star"] = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                idx_src["M_j"] = [(n_src + 4) * n_t + 1, (n_src + 5) * n_t]
                n_src = n_src + 5
            if settings.core:
                if settings.method_core_turb == "GE":
                    idx["mdoti_c"] = [n * n_t + 1, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                    idx["DTt_des_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                    n = n + 5
                    idx_src["mdoti_c_star"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                    idx_src["Tti_c_star"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                    idx_src["Ttj_c_star"] = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                    idx_src["Pti_c_star"] = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                    idx_src["DTt_des_c_star"] = [(n_src + 4) * n_t + 1, (n_src + 5) * n_t]
                    n_src = n_src + 5
                elif settings.method_core_turb == "PW":
                    idx["mdoti_c"] = [n * n_t + 1, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                    idx["rho_te_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                    idx["c_te_c"] = [(n + 5) * n_t + 1, (n + 6) * n_t]
                    idx["rho_ti_c"] = [(n + 6) * n_t + 1, (n + 7) * n_t]
                    idx["c_ti_c"] = [(n + 7) * n_t + 1, (n + 8) * n_t]
                    n = n + 8
                    idx_src["mdoti_c_star"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                    idx_src["Tti_c_star"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                    idx_src["Ttj_c_star"] = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                    idx_src["Pti_c_star"] = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                    idx_src["rho_te_c_star"] = [(n_src + 4) * n_t + 1, (n_src + 5) * n_t]
                    idx_src["c_te_c_star"] = [(n_src + 5) * n_t + 1, (n_src + 6) * n_t]
                    idx_src["rho_ti_c_star"] = [(n_src + 6) * n_t + 1, (n_src + 7) * n_t]
                    idx_src["c_ti_c_star"] = [(n_src + 7) * n_t + 1, (n_src + 8) * n_t]
                    n_src = n_src + 8
            if settings.airframe:
                idx["theta_flaps"] = [n * n_t + 1, (n + 1) * n_t]
                idx["I_landing_gear"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                n = n + 2
                idx_src["theta_flaps"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                idx_src["I_landing_gear"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                n_src = n_src + 2
            if settings.fan_inlet == True or settings.fan_discharge == True:
                idx["DTt_f"] = [n * n_t + 1, (n + 1) * n_t]
                idx["mdot_f"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["N_f"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["A_f"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                idx["d_f"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                idx_src["DTt_f_star"] = [n_src * n_t + 1, (n_src + 1) * n_t]
                idx_src["mdot_f_star"] = [(n_src + 1) * n_t + 1, (n_src + 2) * n_t]
                idx_src["N_f_star"] = [(n_src + 2) * n_t + 1, (n_src + 3) * n_t]
                idx_src["A_f_star"] = [(n_src + 3) * n_t + 1, (n_src + 4) * n_t]
                idx_src["d_f_star"] = [(n_src + 4) * n_t + 1, (n_src + 5) * n_t]

        elif settings.language == 'python':
            idx["x"] = [0 * n_t, 1 * n_t]
            idx["y"] = [1 * n_t, 2 * n_t]
            idx["z"] = [2 * n_t, 3 * n_t]
            idx["alpha"] = [3 * n_t, 4 * n_t]
            idx["gamma"] = [4 * n_t, 5 * n_t]
            idx["t_s"] = [5 * n_t, 6 * n_t]
            idx["rho_0"] = [6 * n_t, 7 * n_t]
            idx["mu_0"] = [7 * n_t, 8 * n_t]
            idx["c_0"] = [8 * n_t, 9 * n_t]
            idx["T_0"] = [9 * n_t, 10 * n_t]
            idx["p_0"] = [10 * n_t, 11 * n_t]
            idx["M_0"] = [11 * n_t, 12 * n_t]
            idx["I_0"] = [12 * n_t, 13 * n_t]
            idx["TS"] = [13 * n_t, 14 * n_t]

            idx_src["TS"] = [0 * n_t, 1 * n_t]
            idx_src["M_0"] = [1 * n_t, 2 * n_t]
            idx_src["c_0"] = [2 * n_t, 3 * n_t]
            idx_src["rho_0"] = [3 * n_t, 4 * n_t]
            idx_src["mu_0"] = [4 * n_t, 5 * n_t]
            idx_src["T_0"] = [5 * n_t, 6 * n_t]
            idx_src["theta"] = [6 * n_t, 7 * n_t]
            idx_src["phi"] = [7 * n_t, 8 * n_t]

            n = 14
            n_src = 8
            if settings.jet_mixing == True and settings.jet_shock == False:
                idx["V_j"]   = [n * n_t, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"]   = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"]  = [(n + 3) * n_t, (n + 4) * n_t]
                n = n + 4
                idx_src["V_j_star"]   = [n_src * n_t, (n_src + 1) * n_t]
                idx_src["rho_j_star"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                idx_src["A_j_star"]   = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                idx_src["Tt_j_star"]  = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                n_src = n_src + 4
            elif settings.jet_shock == True and settings.jet_mixing == False:
                idx["V_j"]  = [n * n_t, (n + 1) * n_t]
                idx["M_j"]       = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"]  = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t, (n + 4) * n_t]
                n = n + 4
                idx_src["V_j_star"] = [n_src * n_t, (n_src + 1) * n_t]
                idx_src["M_j"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                idx_src["A_j_star"] = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                idx_src["Tt_j_star"] = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                n_src = n_src + 4
            elif settings.jet_shock == True and settings.jet_mixing == True:
                idx["V_j"] = [n * n_t, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"] = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t, (n + 4) * n_t]
                idx["M_j"] = [(n + 4) * n_t, (n + 5) * n_t]
                n = n + 5
                idx_src["V_j_star"] = [n_src * n_t, (n_src + 1) * n_t]
                idx_src["rho_j_star"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                idx_src["A_j_star"] = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                idx_src["Tt_j_star"] = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                idx_src["M_j"] = [(n_src + 4) * n_t, (n_src + 5) * n_t]
                n_src = n_src + 5
            if settings.core:
                if settings.method_core_turb == "GE":
                    idx["mdoti_c"] = [n * n_t, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t, (n + 4) * n_t]
                    idx["DTt_des_c"] = [(n + 4) * n_t, (n + 5) * n_t]
                    n = n + 5
                    idx_src["mdoti_c_star"] = [n_src * n_t, (n_src + 1) * n_t]
                    idx_src["Tti_c_star"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                    idx_src["Ttj_c_star"] = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                    idx_src["Pti_c_star"] = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                    idx_src["DTt_des_c_star"] = [(n_src + 4) * n_t, (n_src + 5) * n_t]
                    n_src = n_src + 5
                elif settings.method_core_turb == "PW":
                    idx["mdoti_c"] = [n * n_t, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t, (n + 4) * n_t]
                    idx["rho_te_c"] = [(n + 4) * n_t, (n + 5) * n_t]
                    idx["c_te_c"] = [(n + 5) * n_t, (n + 6) * n_t]
                    idx["rho_ti_c"] = [(n + 6) * n_t, (n + 7) * n_t]
                    idx["c_ti_c"] = [(n + 7) * n_t, (n + 8) * n_t]
                    n = n + 8
                    idx_src["mdoti_c_star"] = [n_src * n_t, (n_src + 1) * n_t]
                    idx_src["Tti_c_star"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                    idx_src["Ttj_c_star"] = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                    idx_src["Pti_c_star"] = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                    idx_src["rho_te_c_star"] = [(n_src + 4) * n_t, (n_src + 5) * n_t]
                    idx_src["c_te_c_star"] = [(n_src + 5) * n_t, (n_src + 6) * n_t]
                    idx_src["rho_ti_c_star"] = [(n_src + 6) * n_t, (n_src + 7) * n_t]
                    idx_src["c_ti_c_star"] = [(n_src + 7) * n_t, (n_src + 8) * n_t]
                    n_src = n_src + 8
            if settings.airframe:
                idx["theta_flaps"] = [n * n_t, (n + 1) * n_t]
                idx["I_landing_gear"] = [(n + 1) * n_t, (n + 2) * n_t]
                n = n + 2
                idx_src["theta_flaps"] = [n_src * n_t, (n_src + 1) * n_t]
                idx_src["I_landing_gear"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                n_src = n_src + 2
            if settings.fan_inlet == True or settings.fan_discharge == True:
                idx["DTt_f"] = [n * n_t, (n + 1) * n_t]
                idx["mdot_f"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["N_f"] = [(n + 2) * n_t, (n + 3) * n_t]
                idx["A_f"] = [(n + 3) * n_t, (n + 4) * n_t]
                idx["d_f"] = [(n + 4) * n_t, (n + 5) * n_t]
                idx_src["DTt_f_star"] = [n_src * n_t, (n_src + 1) * n_t]
                idx_src["mdot_f_star"] = [(n_src + 1) * n_t, (n_src + 2) * n_t]
                idx_src["N_f_star"] = [(n_src + 2) * n_t, (n_src + 3) * n_t]
                idx_src["A_f_star"] = [(n_src + 3) * n_t, (n_src + 4) * n_t]
                idx_src["d_f_star"] = [(n_src + 4) * n_t, (n_src + 5) * n_t]

        return idx, idx_src

    def setup_time_series(self, problem:om.Problem, settings:Settings, ac:Aircraft, n_t:np.int, mode:str, optimization=False) -> None:
        """
        Setup model for computing noise of predefined trajectory time_series.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
        :param n_t: Number of time steps in trajectory
        :type n_t: Dict[str, Any]
        :param optimization: flag to run trajectory optimization
        :type optimization: bool

        :return: None
        """

        # Set solver
        problem.model.linear_solver = om.LinearRunOnce()

        idx, idx_src = Noise.get_indices_noise_input_vector(settings=settings, n_t=n_t)

        if settings.language == 'python':
            problem.model.add_subsystem(name='noise',
                                        subsys=NoiseModel(settings=settings, data=self.data, ac=ac, n_t=n_t, mode=mode), 
                                        promotes_inputs=[],
                                        promotes_outputs=[])
        
        elif settings.language == 'julia':
            problem.model.add_subsystem(name='noise',
                                        subsys=make_component(julia.Noise(settings, self.data, ac, n_t,  idx, idx_src, optimization)),
                                        promotes_inputs=[],
                                        promotes_outputs=[])

        return

    @staticmethod
    def compute_time_series(problem: om.Problem, settings: Settings, path: pd.DataFrame, engine, mode) -> None:
        """
        Compute noise of predefined trajectory time series.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param path: path of trajectory time series
        :type path: pd.DataFrame

        :return: None
        """

        # Run the openMDAO problem setup
        problem.setup()

        # Attach a recorder to the problem to save model data
        if settings.save_results:
            problem.add_recorder(om.SqliteRecorder(settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_file_name))

        if settings.language == 'python':
            # Set variables for engine normalization
            problem['noise.normalize_engine.c_0'] = path['c_0 [m/s]']
            problem['noise.normalize_engine.T_0'] = path['T_0 [K]']
            problem['noise.normalize_engine.p_0'] = path['p_0 [Pa]']
            problem['noise.normalize_engine.rho_0'] = path['rho_0 [kg/m3]']
            if settings.jet_mixing and not settings.jet_shock:
                problem['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                problem['noise.normalize_engine.rho_j'] = engine['Jet rho [kg/m3]']
                problem['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                problem['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            elif settings.jet_shock and not settings.jet_mixing:
                problem['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                problem['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                problem['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            elif settings.jet_shock and settings.jet_mixing:
                problem['noise.normalize_engine.V_j'] = engine['Jet V [m/s]']
                problem['noise.normalize_engine.rho_j'] = engine['Jet rho [kg/m3]']
                problem['noise.normalize_engine.A_j'] = engine['Jet A [m2]']
                problem['noise.normalize_engine.Tt_j'] = engine['Jet Tt [K]']
            if settings.core:
                if settings.method_core_turb == "GE":
                    problem['noise.normalize_engine.mdoti_c'] = engine['Core mdot [kg/s]']
                    problem['noise.normalize_engine.Tti_c'] = engine['Core Tti [K]']
                    problem['noise.normalize_engine.Ttj_c'] = engine['Core Ttj [K]']
                    problem['noise.normalize_engine.Pti_c'] = engine['Core Pt [Pa]']
                    problem['noise.normalize_engine.DTt_des_c'] = engine['Core DT_t [K]']
                elif settings.method_core_turb == "PW":
                    problem['noise.normalize_engine.mdoti_c'] = engine['Core mdot [kg/s]']
                    problem['noise.normalize_engine.Tti_c'] = engine['Core Tti [K]']
                    problem['noise.normalize_engine.Ttj_c'] = engine['Core Ttj [K]']
                    problem['noise.normalize_engine.Pti_c'] = engine['Core Pt [Pa]']
                    problem['noise.normalize_engine.rho_te_c'] = engine['LPT rho_e [kg/m3]']
                    problem['noise.normalize_engine.c_te_c'] = engine['LPT c_e [m/s]']
                    problem['noise.normalize_engine.rho_ti_c'] = engine['HPT rho_i [kg/m3]']
                    problem['noise.normalize_engine.c_ti_c'] = engine['HPT c_i [m/s]']
            if settings.fan_inlet or settings.fan_discharge:
                problem['noise.normalize_engine.DTt_f'] = engine['Fan delta T [K]']
                problem['noise.normalize_engine.mdot_f'] = engine['Fan mdot in [kg/s]']
                problem['noise.normalize_engine.N_f'] = engine['Fan N [rpm]']
                problem['noise.normalize_engine.A_f'] = engine['Fan A [m2]']
                problem['noise.normalize_engine.d_f'] = engine['Fan d [m]']
            # Set variables for geometry
            problem['noise.geometry.x'] = path['X [m]']
            problem['noise.geometry.y'] = path['Y [m]']
            problem['noise.geometry.z'] = path['Z [m]']
            problem['noise.geometry.alpha'] = path['alpha [deg]']
            problem['noise.geometry.gamma'] = path['gamma [deg]']
            problem['noise.geometry.c_0'] = path['c_0 [m/s]']
            problem['noise.geometry.T_0'] = path['T_0 [K]']
            problem['noise.geometry.t_s'] = path['t_source [s]']
            # Set variables for source
            problem['noise.source.TS'] = path['TS [-]']
            problem['noise.source.M_0'] = path['M_0 [-]']
            problem['noise.source.c_0'] = path['c_0 [m/s]']
            problem['noise.source.rho_0'] = path['rho_0 [kg/m3]']
            problem['noise.source.mu_0'] = path['mu_0 [kg/ms]']
            problem['noise.source.T_0'] = path['T_0 [K]']
            if settings.jet_shock and not settings.jet_mixing:
                problem['noise.source.M_j'] = engine['Jet M [-]']
            elif settings.jet_shock and settings.jet_mixing:
                problem['noise.source.M_j'] = engine['Jet M [-]']
            if settings.airframe:
                problem['noise.source.theta_flaps'] = path['Airframe delta_f [deg]']
                problem['noise.source.I_landing_gear'] = path['Airframe LG [-]']
            # Set variables for propagation
            if mode == 'time_series':
                problem['noise.propagation.x'] = path['X [m]']
                problem['noise.propagation.z'] = path['Z [m]']
                problem['noise.propagation.rho_0'] = path['rho_0 [kg/m3]']
                problem['noise.propagation.I_0'] = path['I_0 [kg/m2s]']
            # Set variables for levels
            problem['noise.levels.c_0'] = path['c_0 [m/s]']
            problem['noise.levels.rho_0'] = path['rho_0 [kg/m3]']

        elif settings.language == 'julia':
            # Set path parameters
            problem['noise.x'] = path['X [m]']
            problem['noise.y'] = path['Y [m]']
            problem['noise.z'] = path['Z [m]']
            problem['noise.alpha'] = path['alpha [deg]']
            problem['noise.gamma'] = path['gamma [deg]']
            problem['noise.t_s'] = path['t_source [s]']
            problem['noise.rho_0'] = path['rho_0 [kg/m3]']
            problem['noise.c_0'] = path['c_0 [m/s]']
            problem['noise.mu_0'] = path['mu_0 [kg/ms]']
            problem['noise.T_0'] = path['T_0 [K]']
            problem['noise.p_0'] = path['p_0 [Pa]']
            problem['noise.M_0'] = path['M_0 [-]']
            problem['noise.I_0'] = path['I_0 [kg/m2s]']
            problem['noise.TS'] = path['TS [-]']

            if settings.jet_mixing and not settings.jet_shock:
                problem['noise.V_j'] = engine['Jet V [m/s]']
                problem['noise.rho_j'] = engine['Jet rho [kg/m3]']
                problem['noise.A_j'] = engine['Jet A [m2]']
                problem['noise.Tt_j'] = engine['Jet Tt [K]']
            elif settings.jet_shock and not settings.jet_mixing:
                problem['noise.V_j'] = engine['Jet V [m/s]']
                problem['noise.M_j'] = engine['Jet M [-]']
                problem['noise.A_j'] = engine['Jet A [m2]']
                problem['noise.Tt_j'] = engine['Jet Tt [K]']
            elif settings.jet_shock and settings.jet_mixing:
                problem['noise.V_j'] = engine['Jet V [m/s]']
                problem['noise.rho_j'] = engine['Jet rho [kg/m3]']
                problem['noise.A_j'] = engine['Jet A [m2]']
                problem['noise.Tt_j'] = engine['Jet Tt [K]']
                problem['noise.M_j'] = engine['Jet M [-]']

            if settings.core:
                if settings.method_core_turb == "GE":
                    problem['noise.mdoti_c'] = engine['Core mdot [kg/s]']
                    problem['noise.Tti_c'] = engine['Core Tti [K]']
                    problem['noise.Ttj_c'] = engine['Core Ttj [K]']
                    problem['noise.Pti_c'] = engine['Core Pt [Pa]']
                    problem['noise.DTt_des_c'] = engine['Core DT_t [K]']
                elif settings.method_core_turb == "PW":
                    problem['noise.mdoti_c'] = engine['Core mdot [kg/s]']
                    problem['noise.Tti_c'] = engine['Core Tti [K]']
                    problem['noise.Ttj_c'] = engine['Core Ttj [K]']
                    problem['noise.Pti_c'] = engine['Core Pt [Pa]']
                    problem['noise.rho_te_c'] = engine['LPT rho_e [kg/m3]']
                    problem['noise.c_te_c'] = engine['LPT c_e [m/s]']
                    problem['noise.rho_ti_c'] = engine['HPT rho_i [kg/m3]']
                    problem['noise.c_ti_c'] = engine['HPT c_i [m/s]']

            if settings.airframe:
                problem['noise.theta_flaps'] = path['Airframe delta_f [deg]']
                problem['noise.I_landing_gear'] = path['Airframe LG [-]']

            if settings.fan_inlet or settings.fan_discharge:
                problem['noise.DTt_f'] = engine['Fan delta T [K]']
                problem['noise.mdot_f'] = engine['Fan mdot in [kg/s]']
                problem['noise.N_f'] = engine['Fan N [rpm]']
                problem['noise.A_f'] = engine['Fan A [m2]']
                problem['noise.d_f'] = engine['Fan d [m]']

        # Run noise module
        dm.run_problem(problem, run_driver=False, simulate=False, solution_record_file=settings.pyNA_directory + '/cases/' + settings.case_name + '/dymos_solution.db')

        # Save the results
        if settings.save_results:
            problem.record('time_series')

        return

    def setup_trajectory_noise(self, problem: om.Problem, settings: Settings, ac: Aircraft, n_t:np.int, optimization=False) -> None:
        """
        Setup model for computing noise along computed trajectory.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
        :param n_t: Number of time steps in trajectory
        :type n_t: np.int
        :param optimization: flag to run trajectory optimization
        :type optimization: bool

        :return: None
        """

        idx, idx_src = Noise.get_indices_noise_input_vector(settings=settings, n_t=n_t)

        problem.model.add_subsystem(name='noise',
                                    subsys=make_component(julia.Noise(settings, self.data, ac, n_t,  idx, idx_src, optimization)),
                                    promotes_inputs=[],
                                    promotes_outputs=[])

        # Create connections from trajectory group
        problem.model.connect('trajectory.x', 'noise.x')
        problem.model.connect('trajectory.y', 'noise.y')
        problem.model.connect('trajectory.z', 'noise.z')
        problem.model.connect('trajectory.alpha', 'noise.alpha')
        problem.model.connect('trajectory.gamma', 'noise.gamma')
        problem.model.connect('trajectory.t_s', 'noise.t_s')
        problem.model.connect('trajectory.TS', 'noise.TS')
        problem.model.connect('trajectory.M_0', 'noise.M_0')
        problem.model.connect('trajectory.p_0', 'noise.p_0')
        problem.model.connect('trajectory.c_0', 'noise.c_0')
        problem.model.connect('trajectory.T_0', 'noise.T_0')
        problem.model.connect('trajectory.rho_0', 'noise.rho_0')
        problem.model.connect('trajectory.mu_0', 'noise.mu_0')
        problem.model.connect('trajectory.I_0', 'noise.I_0')
        if settings.airframe:
            problem.model.connect('trajectory.I_landing_gear', 'noise.I_landing_gear')
            problem.model.connect('trajectory.theta_flaps', 'noise.theta_flaps')

        # Create connections from engine component
        if settings.jet_mixing and settings.jet_shock == False:
            problem.model.connect('engine.V_j', 'noise.V_j')
            problem.model.connect('engine.rho_j', 'noise.rho_j')
            problem.model.connect('engine.A_j', 'noise.A_j')
            problem.model.connect('engine.Tt_j', 'noise.Tt_j')
        elif settings.jet_shock and settings.jet_mixing == False:
            problem.model.connect('engine.V_j', 'noise.V_j')
            problem.model.connect('engine.A_j', 'noise.A_j')
            problem.model.connect('engine.Tt_j', 'noise.Tt_j')
            problem.model.connect('engine.M_j', 'noise.M_j')
        elif settings.jet_shock and settings.jet_mixing:
            problem.model.connect('engine.V_j', 'noise.V_j')
            problem.model.connect('engine.rho_j', 'noise.rho_j')
            problem.model.connect('engine.A_j', 'noise.A_j')
            problem.model.connect('engine.Tt_j', 'noise.Tt_j')
            problem.model.connect('engine.M_j', 'noise.M_j')
        if settings.core:
            if settings.method_core_turb == 'GE':
                problem.model.connect('engine.mdoti_c', 'noise.mdoti_c')
                problem.model.connect('engine.Tti_c', 'noise.Tti_c')
                problem.model.connect('engine.Ttj_c', 'noise.Ttj_c')
                problem.model.connect('engine.Pti_c', 'noise.Pti_c')
                problem.model.connect('engine.DTt_des_c', 'noise.DTt_des_c')
            elif settings.method_core_turb == 'PW':
                problem.model.connect('engine.mdoti_c', 'noise.mdoti_c')
                problem.model.connect('engine.Tti_c', 'noise.Tti_c')
                problem.model.connect('engine.Ttj_c', 'noise.Ttj_c')
                problem.model.connect('engine.Pti_c', 'noise.Pti_c')
                problem.model.connect('engine.rho_te_c', 'noise.rho_te_c')
                problem.model.connect('engine.c_te_c', 'noise.c_te_c')
                problem.model.connect('engine.rho_ti_c', 'noise.rho_ti_c')
                problem.model.connect('engine.c_ti_c', 'noise.c_ti_c')
        if settings.fan_inlet or settings.fan_discharge:
            problem.model.connect('engine.DTt_f', 'noise.DTt_f')
            problem.model.connect('engine.mdot_f', 'noise.mdot_f')
            problem.model.connect('engine.N_f', 'noise.N_f')
            problem.model.connect('engine.A_f', 'noise.A_f')
            problem.model.connect('engine.d_f', 'noise.d_f')

        # Add objective for trajectory model
        if optimization:
            problem.model.add_objective('noise.'+settings.levels_int_metric, ref=1.)
        return None


























    def setup_source_distribution(self, problem: om.Problem, settings: Settings, ac: Aircraft, engine:Engine, comp: str, time_step: np.int64):
        """
        Setup model for computing noise source directional and spectral distribution.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
        :param engine: engine parameters
        :type engine: Engine
        :param comp: noise source component
        :type comp: str
        :param time_step: Time step in predefined trajectory at which to compute the noise source distribution.
        :type time_step: np.int64

        :return: None
        """

        # Reset component flags
        settings.all_sources = False
        settings.fan_inlet = False
        settings.fan_discharge = False
        settings.core = False
        settings.jet_mixing = False
        settings.jet_shock = False
        settings.airframe = False

        # Enable component flag
        if comp == 'fan_inlet':
            settings.fan_inlet = True
        elif comp == 'fan_discharge':
            settings.fan_discharge = True
        elif comp == 'core':
            settings.core = True
        elif comp == 'jet_mixing':
            settings.jet_mixing = True
        elif comp == 'airframe':
            settings.airframe = True

        # Add the subsystems to the openmdao problem
        problem.model.add_subsystem(name='engine', subsys=NormalizeEngineData(settings=settings, engine=engine.time_series['source'], n_t=np.size(time_step), time_step=time_step))
        problem.model.add_subsystem(name='source',
                                    subsys=Source(settings=settings, data=self.data, ac=ac, n_t=np.size(time_step), time_step=time_step),
                                    promotes_inputs=[],
                                    promotes_outputs=[])
        problem.model.add_subsystem(name='levels',
                                    subsys=Levels(settings=settings, data=self.data, n_t=np.size(time_step), time_step=time_step),
                                    promotes_inputs=[],
                                    promotes_outputs=[])

        # Create connections between the subsystems
        if settings.jet_mixing and settings.jet_shock == False:
            problem.model.connect('engine.V_j', 'source.V_j')
            problem.model.connect('engine.rho_j', 'source.rho_j')
            problem.model.connect('engine.A_j', 'source.A_j')
            problem.model.connect('engine.Tt_j', 'source.Tt_j')
        elif settings.jet_shock and settings.jet_mixing == False:
            problem.model.connect('engine.V_j', 'source.V_j')
            problem.model.connect('engine.A_j', 'source.A_j')
            problem.model.connect('engine.Tt_j', 'source.Tt_j')
            problem.model.connect('engine.M_j', 'source.M_j')
        elif settings.jet_shock and settings.jet_mixing:
            problem.model.connect('engine.V_j', 'source.V_j')
            problem.model.connect('engine.rho_j', 'source.rho_j')
            problem.model.connect('engine.A_j', 'source.A_j')
            problem.model.connect('engine.Tt_j', 'source.Tt_j')
            problem.model.connect('engine.M_j', 'source.M_j')
        if settings.core:
            if settings.method_core_turb == 'GE':
                problem.model.connect('engine.mdoti_c', 'source.mdoti_c')
                problem.model.connect('engine.Tti_c', 'source.Tti_c')
                problem.model.connect('engine.Ttj_c', 'source.Ttj_c')
                problem.model.connect('engine.Pti_c', 'source.Pti_c')
                problem.model.connect('engine.DTt_des_c', 'source.DTt_des_c')
            elif settings.method_core_turb == 'PW':
                problem.model.connect('engine.mdoti_c', 'source.mdoti_c')
                problem.model.connect('engine.Tti_c', 'source.Tti_c')
                problem.model.connect('engine.Ttj_c', 'source.Ttj_c')
                problem.model.connect('engine.Pti_c', 'source.Pti_c')
                problem.model.connect('engine.rho_te_c', 'source.rho_te_c')
                problem.model.connect('engine.c_te_c', 'source.c_te_c')
                problem.model.connect('engine.rho_ti_c', 'source.rho_ti_c')
                problem.model.connect('engine.c_ti_c', 'source.c_ti_c')
        if settings.fan_inlet or settings.fan_discharge:
            problem.model.connect('engine.DTt_f', 'source.DTt_f')
            problem.model.connect('engine.mdot_f', 'source.mdot_f')
            problem.model.connect('engine.N_f', 'source.N_f')
            problem.model.connect('engine.A_f', 'source.A_f')
            problem.model.connect('engine.d_f', 'source.d_f')
        problem.model.connect('source.msap_source', 'levels.msap_prop')

        return None
    @staticmethod
    def compute_source_distribution(problem: om.Problem, settings: Settings, path: pd.DataFrame, time_step: np.int64, theta: np.float64) -> None:
        """
        Compute noise source directional and spectral distribution.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param path: path of trajectory time_series
        :type path: pd.DataFrame
        :param time_step: Time step in predefined trajectory at which to compute the noise source distribution.
        :type time_step: np.int64
        :param theta: polar directivity angle
        :type theta: np.float64

        :return: None
        """

        # Run setup
        problem.setup()

        # Attach a recorder to the problem to save model data
        if settings.save_results:
            problem.add_recorder(om.SqliteRecorder(settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_file_name))

        # Set initial values
        path_time_series = path['source']
        problem.set_val('engine.c_0', path_time_series['c_0 [m/s]'][time_step])
        problem.set_val('engine.T_0', path_time_series['T_0 [K]'][time_step])
        problem.set_val('engine.p_0', path_time_series['p_0 [Pa]'][time_step])
        problem.set_val('engine.rho_0', path_time_series['rho_0 [kg/m3]'][time_step])

        problem.set_val('source.TS', path_time_series['TS [-]'][time_step])
        problem.set_val('source.M_0', path_time_series['M_0 [-]'][time_step])
        problem.set_val('source.c_0', path_time_series['c_0 [m/s]'][time_step])
        problem.set_val('source.T_0', path_time_series['T_0 [K]'][time_step])
        problem.set_val('source.mu_0', path_time_series['mu_0 [kg/ms]'][time_step])
        problem.set_val('source.rho_0', path_time_series['rho_0 [kg/m3]'][time_step])

        problem.set_val('levels.c_0', path_time_series['c_0 [m/s]'][time_step])
        problem.set_val('levels.rho_0', path_time_series['rho_0 [kg/m3]'][time_step])

        if settings.airframe:
            problem.set_val('source.theta_flaps', path_time_series['Airframe delta_f [deg]'][time_step])
            problem.set_val('source.I_landing_gear', path_time_series['Airframe LG [-]'][time_step])

        problem.set_val('source.theta', theta * np.ones(np.size(path_time_series['t_source [s]']))[time_step])
        problem.set_val('source.phi', np.zeros(np.size(path_time_series['t_source [s]']))[time_step])

        # Run model
        problem.run_model()

        # Save the results
        if settings.save_results:
            problem.record(settings.observer)

        return None

    @staticmethod
    def compute_epnl(t_o: np.ndarray, pnlt: np.ndarray, C: np.ndarray = None)->np.float64:
        """
        Compute EPNL from time series.

        :param t_o: observer time [s]
        :type t_o: np.ndarray
        :param pnlt: tone-corrected perceived noise level [dB]
        :type pnlt: np.ndarray
        :param C: tone corrections for pnlt [dB]
        :type C: np.ndarray

        :return: epnl
        :rtype: np.flaot64
        """

        # Initialize settings
        settings = dict()
        settings.levels_int_metric = 'epnl'
        settings.N_f = 24
        settings.n_t = np.size(t_o)
        if np.size(C):
            settings.bandshare = True
        else:
            settings.bandshare = True

        # Initialize problem
        problem = om.Problem()
        problem.model.add_subsystem(name='l',
                           subsys=LevelsInt(settings=settings),
                           promotes_inputs=[],
                           promotes_outputs=[])
        problem.setup(force_alloc_complex=False)

        # Set values
        problem.set_val('l.pnlt', pnlt)
        problem.set_val('l.t_o', t_o)
        if np.size(C):
            problem.set_val('l.C', C)

        # Run model
        problem.run_model()
        epnl = problem.get_val('l.epnl')

        return epnl





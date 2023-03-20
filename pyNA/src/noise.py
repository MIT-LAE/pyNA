import pandas as pd
from typing import Union
import openmdao.api as om
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory import Trajectory
from pyNA.src.time_history import TimeHistory
from pyNA.src.noise_model.tables import Tables
from pyNA.src.noise_model.python.utils.get_frequency_bands import get_frequency_bands
from pyNA.src.noise_model.python.utils.get_frequency_subbands import get_frequency_subbands
from pyNA.src.noise_model.python.noise_model import NoiseModel


class Noise:
    
    """
    
    Parameters
    ----------
    problem : om.Problem
        _
    settings : dict
        _
    aircraft : Aircraft
        _
    
    Attributes
    ----------
    tables : Tables
        _
    f : np.ndarray
        _
    f_sb : np.ndarray
        _

    """

    def __init__(self, problem: om.Problem, settings: dict, aircraft: Aircraft, trajectory: Union[Trajectory, TimeHistory]) -> None:
        
        self.problem = problem
        self.settings = settings
        self.aircraft = aircraft
        self.trajectory = trajectory

        self.tables = Tables(settings=settings)
        self.f = get_frequency_bands(n_frequency_bands=settings['n_frequency_bands'])
        self.f_sb = get_frequency_subbands(f=self.f, n_frequency_subbands=settings['n_frequency_subbands'])

    def connect(self, optimization=False):
        
        """
        
        Parameters
        ----------
        problem : om.Problem
            _
        settings : dict
            pyna settings
        aircraft : Aircraft
            _
        n_t : int
            Number of time steps

        """

        # Set solver
        self.problem.model.linear_solver = om.LinearRunOnce()

        promote_lst = ['x', 'y', 'z', 'alpha', 'gamma', 't_s', 'tau', 
                       'M_0', 'c_0', 'T_0', 'rho_0', 'P_0', 'mu_0', 'I_0', 
                       'fan_DTt', 'fan_mdot', 'fan_N', 
                       'core_mdot', 'core_Tt_i', 'core_Tt_j', 'core_Pt_i', 'turb_DTt_des', 'turb_rho_e', 'turb_c_e', 'turb_rho_i', 'turb_c_i', 
                       'jet_V', 'jet_rho', 'jet_A', 'jet_Tt', 'jet_M', 
                       'theta_flaps', 'I_lg']

        self.problem.model.add_subsystem(name='noise',
                                 subsys=NoiseModel(settings=self.settings, aircraft=self.aircraft, tables=self.tables, f=self.f, f_sb=self.f_sb, n_t=self.trajectory.n_t, optimization=optimization), 
                                 promotes_inputs=promote_lst,
                                 promotes_outputs=[])

        return

    # def compute_noise_level():
    #     pass

    # def compute_epnl_table():
    #     pass

    # def compute_noise_contours():
    #     pass

    # def compute_noise_contour_area():
    #     pass

    # def plot_noise_level():
    #     pass

    # def plot_noise_contours():
    #     pass

    # def plot_noise_source_distribution():
    #     pass
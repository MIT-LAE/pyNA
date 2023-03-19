import numpy as np
import pandas as pd
import openmdao.api as om
from pyNA.src.trajectory_model.time_history.trajectory_data import TrajectoryData
import pyNA


class TimeHistory:
    
    """
    
    Attributes
    ----------
    data : pd.DataFrame
        _
    vars : list
        _
    var_units : list
        _
    n_t : int
        _

    """

    def __init__(self) -> None:
        
        self.data = pd.DataFrame()
        self.vars = list()
        self.var_units = dict()
        self.n_t = 1

    def load_data(self, settings) -> None:
        
        """
        Load engine and trajectory time history from .csv file.

        Parameters
        ----------
        settings : dict
            _

        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        try:
            self.data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/' + settings['time_history_file_name'])
        except:
            raise ValueError(settings['time_history_file_name'] + ' file not found at ' + pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/')
        
        # Compute number of time steps in the time history
        self.n_t = np.size(self.data['t_s [s]'])

        for col in self.data.columns:
            var, unit = col.split(' ')
            unit = unit.replace('[','').replace(']','')
            self.vars.append(var)
            if unit == '-':
                self.var_units[var] = None
            else:
                self.var_units[var] = unit

        return None

    def connect(self, problem: om.Problem) -> None:
        
        """
        
        Parameters
        ----------
        problem : om.Problem
            _
        num_nodes : int
            _
        vars : list
            _
        var_units : list
            _

        """

        problem.model.add_subsystem(name='trajectory', 
                                    subsys=TrajectoryData(num_nodes=self.n_t, vars=self.vars, var_units=self.var_units),
                                    promotes_outputs=self.vars)

        return None

    def set_initial_conditions(self, problem: om.Problem, settings: dict) -> None:

        """
        
        Parameters
        ----------
        problem : 
            _
        settings : pyNA settings
            _
        
        """
        
        # Trajectory variables
        problem.model.set_val('trajectory.t_s', self.data['t_s [s]'])
        problem.model.set_val('trajectory.x', self.data['x [m]'])
        problem.model.set_val('trajectory.y', self.data['y [m]'])
        problem.model.set_val('trajectory.z', self.data['z [m]'])
        problem.model.set_val('trajectory.v', self.data['v [m/s]'])
        problem.model.set_val('trajectory.alpha', self.data['alpha [deg]'])
        problem.model.set_val('trajectory.gamma', self.data['gamma [deg]'])
        problem.model.set_val('trajectory.F_n', self.data['F_n [N]'])
        problem.model.set_val('trajectory.tau', self.data['tau [-]'])
        problem.model.set_val('trajectory.M_0', self.data['M_0 [-]'])
        problem.model.set_val('trajectory.c_0', self.data['c_0 [m/s]'])
        problem.model.set_val('trajectory.T_0', self.data['T_0 [K]'])
        problem.model.set_val('trajectory.p_0', self.data['p_0 [Pa]'])
        problem.model.set_val('trajectory.rho_0', self.data['rho_0 [kg/m**3]'])
        problem.model.set_val('trajectory.mu_0', self.data['mu_0 [kg/m/s]'])
        problem.model.set_val('trajectory.I_0', self.data['I_0 [kg/m**2/s]'])
        
        if settings['airframe_source']:
            problem.model.set_val('trajectory.theta_flaps', self.data['theta_flaps [deg]'])
            problem.model.set_val('trajectory.I_lg', self.data['I_lg [-]'])

        # Engine variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            problem.model.set_val('trajectory.jet_V', self.data['jet_V [m/s]'])
            problem.model.set_val('trajectory.jet_rho', self.data['jet_rho [kg/m**3]'])
            problem.model.set_val('trajectory.jet_A', self.data['jet_A [m2]'])
            problem.model.set_val('trajectory.jet_Tt', self.data['jet_Tt [K]'])
        elif not settings['jet_mixing_source'] and settings['jet_shock_source']:
            problem.model.set_val('trajectory.jet_V', self.data['jet_V [m/s]'])
            problem.model.set_val('trajectory.jet_A', self.data['jet_A [m**2]'])
            problem.model.set_val('trajectory.jet_Tt', self.data['jet_Tt [K]'])
            problem.model.set_val('trajectory.jet_M', self.data['jet_M [-]'])
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            problem.model.set_val('trajectory.jet_V', self.data['jet_V [m/s]'])
            problem.model.set_val('trajectory.jet_rho', self.data['jet_rho [kg/m**3]'])
            problem.model.set_val('trajectory.jet_A', self.data['jet_A [m**2]'])
            problem.model.set_val('trajectory.jet_Tt', self.data['jet_Tt [K]'])
            problem.model.set_val('trajectory.jet_M', self.data['jet_M [-]'])

        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == "ge":
                problem.model.set_val('trajectory.core_mdot', self.data['core_mdot [kg/s]'])
                problem.model.set_val('trajectory.core_Tt_i', self.data['core_Tt_i [K]'])
                problem.model.set_val('trajectory.core_Tt_j', self.data['core_Tt_j [K]'])
                problem.model.set_val('trajectory.core_Pt_i', self.data['core_Pt_i [Pa]'])
                problem.model.set_val('trajectory.turb_DTt_des', self.data['turb_DTt_des [K]'])
            elif settings['core_turbine_attenuation_method'] == "pw":
                problem.model.set_val('trajectory.core_mdot', self.data['core_mdot [kg/s]'])
                problem.model.set_val('trajectory.core_Tt_i', self.data['core_Tt_i [K]'])
                problem.model.set_val('trajectory.core_Tt_j', self.data['core_Tt_j [K]'])
                problem.model.set_val('trajectory.core_Pt_i', self.data['core_Pt_i [Pa]'])
                problem.model.set_val('trajectory.turb_rho_i', self.data['turb_rho_i [kg/m**3]'])
                problem.model.set_val('trajectory.turb_c_i', self.data['turb_c_i [m/s]'])
                problem.model.set_val('trajectory.turb_rho_e', self.data['turb_rho_e [kg/m**3]'])
                problem.model.set_val('trajectory.turb_c_e', self.data['turb_c_e [m/s]'])
                
        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            problem.model.set_val('trajectory.fan_DTt', self.data['fan_DTt [K]'])
            problem.model.set_val('trajectory.fan_mdot', self.data['fan_mdot [kg/s]'])
            problem.model.set_val('trajectory.fan_N', self.data['fan_N [rpm]'])
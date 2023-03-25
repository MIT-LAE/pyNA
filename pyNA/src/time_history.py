import numpy as np
import pandas as pd
import openmdao.api as om
from pyNA.src.trajectory_model.time_history.trajectory_data import TrajectoryData
import pyNA
import pdb


class TimeHistory:
    
    """
     
    Parameters
    ----------
    problem : om.Problem
        _
    settings : dict
        pyna settings

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

    def __init__(self, problem: om.Problem, settings : dict) -> None:
        
        self.problem = problem
        self.settings = settings
        self.vars = list()
        self.var_units = dict()
        self.n_t = 1
        TimeHistory.get_var(self)

    def get_var(self) -> None:

        self.vars.append('t_s'); self.var_units['t_s'] = 's'
        self.vars.append('x'); self.var_units['x'] = 'm'
        self.vars.append('y'); self.var_units['y'] = 'm'
        self.vars.append('z'); self.var_units['z'] = 'm'
        self.vars.append('alpha'); self.var_units['alpha'] = 'deg'
        self.vars.append('gamma'); self.var_units['gamma'] = 'deg'
        self.vars.append('F_n'); self.var_units['F_n'] = 'N'
        self.vars.append('tau'); self.var_units['tau'] = None
        self.vars.append('v'); self.var_units['v'] = 'm/s'
        self.vars.append('M_0'); self.var_units['M_0'] = None
        self.vars.append('c_0'); self.var_units['c_0'] = 'm/s'
        self.vars.append('T_0'); self.var_units['T_0'] = 'K'
        self.vars.append('P_0'); self.var_units['P_0'] = 'Pa'
        self.vars.append('rho_0'); self.var_units['rho_0'] = 'kg/m**3'
        self.vars.append('mu_0'); self.var_units['mu_0'] = 'kg/m/s'
        self.vars.append('I_0'); self.var_units['I_0'] = 'kg/m**2/s'
        self.vars.append('theta_flaps'); self.var_units['theta_flaps'] = 'deg'
        self.vars.append('I_landing_gear'); self.var_units['I_landing_gear'] = None
        self.vars.append('jet_V'); self.var_units['jet_V'] = 'm/s'
        self.vars.append('jet_A'); self.var_units['jet_A'] = 'm**2'
        self.vars.append('jet_rho'); self.var_units['jet_rho'] = 'kg/m**3'
        self.vars.append('jet_Tt'); self.var_units['jet_Tt'] = 'K'
        self.vars.append('jet_M'); self.var_units['jet_M'] = None
        self.vars.append('core_mdot'); self.var_units['core_mdot'] = 'kg/s'
        self.vars.append('core_Pt_i'); self.var_units['core_Pt_i'] = 'Pa'
        self.vars.append('core_Tt_i'); self.var_units['core_Tt_i'] = 'K'
        self.vars.append('core_Tt_j'); self.var_units['core_Tt_j'] = 'K'
        self.vars.append('turb_DTt_des'); self.var_units['turb_DTt_des'] = 'K'
        self.vars.append('turb_rho_i'); self.var_units['turb_rho_i'] = 'kg/m**3'
        self.vars.append('turb_c_i'); self.var_units['turb_c_i'] = 'm/s'
        self.vars.append('turb_rho_e'); self.var_units['turb_rho_e'] = 'kg/m**3'
        self.vars.append('turb_c_e'); self.var_units['turb_c_e'] = 'm/s'
        self.vars.append('fan_DTt'); self.var_units['fan_DTt'] = 'K'
        self.vars.append('fan_mdot'); self.var_units['fan_mdot'] = 'kg/s'
        self.vars.append('fan_N'); self.var_units['fan_N'] = 'rpm'

    def load_data_from_file(self) -> None:
        
        """
        Load engine and trajectory time history from .csv file.

        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        try:
            self.data_file = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/cases/' + self.settings['case_name'] + '/trajectory/' + self.settings['output_directory_name'] + '/' + self.settings['time_history_file_name'])
            self._data_input_mode = 'file'
        except:
            raise ValueError(self.settings['time_history_file_name'] + ' file not found at ' + pyNA.__path__.__dict__["_path"][0] + '/cases/' + self.settings['case_name'] + '/trajectory/' + self.settings['output_directory_name'] + '/')
        
        # Compute number of time steps in the time history
        self.n_t = np.size(self.data_file['t_s [s]'])

        return None
    
    def load_data_from_problem(self, problem: om.Problem):

        """
        
        """

        self.data_problem = problem
        self._data_input_mode = 'problem'

        # Compute number of time steps in the time history
        self.n_t = np.size(self.data_problem.get_val('trajectory.t_s'))

        
        return

    def connect(self) -> None:
        
        """
        
        Parameters
        ----------
        problem : om.Problem
            _

        """




        self.problem.model.add_subsystem(name='trajectory', 
                                         subsys=TrajectoryData(num_nodes=self.n_t, vars=self.vars, var_units=self.var_units),
                                         promotes_outputs=self.vars)

        return None

    def set_initial_conditions(self) -> None:

        """
                
        """
        
        if self._data_input_mode == 'file':
            
            self.problem.model.set_val('trajectory.t_s', self.data_file['t_s [s]'])
            self.problem.model.set_val('trajectory.x', self.data_file['x [m]'])
            self.problem.model.set_val('trajectory.y', self.data_file['y [m]'])
            self.problem.model.set_val('trajectory.z', self.data_file['z [m]'])
            self.problem.model.set_val('trajectory.v', self.data_file['v [m/s]'])
            self.problem.model.set_val('trajectory.alpha', self.data_file['alpha [deg]'])
            self.problem.model.set_val('trajectory.gamma', self.data_file['gamma [deg]'])
            self.problem.model.set_val('trajectory.F_n', self.data_file['F_n [N]'])
            self.problem.model.set_val('trajectory.tau', self.data_file['tau [-]'])
            self.problem.model.set_val('trajectory.M_0', self.data_file['M_0 [-]'])
            self.problem.model.set_val('trajectory.c_0', self.data_file['c_0 [m/s]'])
            self.problem.model.set_val('trajectory.T_0', self.data_file['T_0 [K]'])
            self.problem.model.set_val('trajectory.P_0', self.data_file['P_0 [Pa]'])
            self.problem.model.set_val('trajectory.rho_0', self.data_file['rho_0 [kg/m**3]'])
            self.problem.model.set_val('trajectory.mu_0', self.data_file['mu_0 [kg/m/s]'])
            self.problem.model.set_val('trajectory.I_0', self.data_file['I_0 [kg/m**2/s]'])
            
            if self.settings['airframe_source']:
                self.problem.model.set_val('trajectory.theta_flaps', self.data_file['theta_flaps [deg]'])
                self.problem.model.set_val('trajectory.I_landing_gear', self.data_file['I_landing_gear [-]'])

            if self.settings['jet_mixing_source'] and not self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_file['jet_V [m/s]'])
                self.problem.model.set_val('trajectory.jet_rho', self.data_file['jet_rho [kg/m**3]'])
                self.problem.model.set_val('trajectory.jet_A', self.data_file['jet_A [m**2]'])
                self.problem.model.set_val('trajectory.jet_Tt', self.data_file['jet_Tt [K]'])
            elif not self.settings['jet_mixing_source'] and self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_file['jet_V [m/s]'])
                self.problem.model.set_val('trajectory.jet_A', self.data_file['jet_A [m**2]'])
                self.problem.model.set_val('trajectory.jet_Tt', self.data_file['jet_Tt [K]'])
                self.problem.model.set_val('trajectory.jet_M', self.data_file['jet_M [-]'])
            elif self.settings['jet_mixing_source'] and self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_file['jet_V [m/s]'])
                self.problem.model.set_val('trajectory.jet_rho', self.data_file['jet_rho [kg/m**3]'])
                self.problem.model.set_val('trajectory.jet_A', self.data_file['jet_A [m**2]'])
                self.problem.model.set_val('trajectory.jet_Tt', self.data_file['jet_Tt [K]'])
                self.problem.model.set_val('trajectory.jet_M', self.data_file['jet_M [-]'])

            if self.settings['core_source']:
                if self.settings['core_turbine_attenuation_method'] == "ge":
                    self.problem.model.set_val('trajectory.core_mdot', self.data_file['core_mdot [kg/s]'])
                    self.problem.model.set_val('trajectory.core_Tt_i', self.data_file['core_Tt_i [K]'])
                    self.problem.model.set_val('trajectory.core_Tt_j', self.data_file['core_Tt_j [K]'])
                    self.problem.model.set_val('trajectory.core_Pt_i', self.data_file['core_Pt_i [Pa]'])
                    self.problem.model.set_val('trajectory.turb_DTt_des', self.data_file['turb_DTt_des [K]'])
                elif self.settings['core_turbine_attenuation_method'] == "pw":
                    self.problem.model.set_val('trajectory.core_mdot', self.data_file['core_mdot [kg/s]'])
                    self.problem.model.set_val('trajectory.core_Tt_i', self.data_file['core_Tt_i [K]'])
                    self.problem.model.set_val('trajectory.core_Tt_j', self.data_file['core_Tt_j [K]'])
                    self.problem.model.set_val('trajectory.core_Pt_i', self.data_file['core_Pt_i [Pa]'])
                    self.problem.model.set_val('trajectory.turb_rho_i', self.data_file['turb_rho_i [kg/m**3]'])
                    self.problem.model.set_val('trajectory.turb_c_i', self.data_file['turb_c_i [m/s]'])
                    self.problem.model.set_val('trajectory.turb_rho_e', self.data_file['turb_rho_e [kg/m**3]'])
                    self.problem.model.set_val('trajectory.turb_c_e', self.data_file['turb_c_e [m/s]'])
                    
            if self.settings['fan_inlet_source'] or self.settings['fan_discharge_source']:
                self.problem.model.set_val('trajectory.fan_DTt', self.data_file['fan_DTt [K]'])
                self.problem.model.set_val('trajectory.fan_mdot', self.data_file['fan_mdot [kg/s]'])
                self.problem.model.set_val('trajectory.fan_N', self.data_file['fan_N [rpm]'])

        elif self._data_input_mode == 'problem':
            
            self.problem.model.set_val('trajectory.t_s', self.data_problem.get_val('trajectory.t_s'))
            self.problem.model.set_val('trajectory.x', self.data_problem.get_val('trajectory.x'))
            self.problem.model.set_val('trajectory.y', self.data_problem.get_val('trajectory.y'))
            self.problem.model.set_val('trajectory.z', self.data_problem.get_val('trajectory.z'))
            self.problem.model.set_val('trajectory.v', self.data_problem.get_val('trajectory.v'))
            self.problem.model.set_val('trajectory.alpha', self.data_problem.get_val('trajectory.alpha'))
            self.problem.model.set_val('trajectory.gamma', self.data_problem.get_val('trajectory.gamma'))
            self.problem.model.set_val('trajectory.F_n', self.data_problem.get_val('trajectory.F_n'))
            self.problem.model.set_val('trajectory.tau', self.data_problem.get_val('trajectory.tau'))
            self.problem.model.set_val('trajectory.M_0', self.data_problem.get_val('trajectory.M_0'))
            self.problem.model.set_val('trajectory.c_0', self.data_problem.get_val('trajectory.c_0'))
            self.problem.model.set_val('trajectory.T_0', self.data_problem.get_val('trajectory.T_0'))
            self.problem.model.set_val('trajectory.P_0', self.data_problem.get_val('trajectory.P_0'))
            self.problem.model.set_val('trajectory.rho_0', self.data_problem.get_val('trajectory.rho_0'))
            self.problem.model.set_val('trajectory.mu_0', self.data_problem.get_val('trajectory.mu_0'))
            self.problem.model.set_val('trajectory.I_0', self.data_problem.get_val('trajectory.I_0'))
            
            if self.settings['airframe_source']:
                self.problem.model.set_val('trajectory.theta_flaps', self.data_problem.get_val('trajectory.theta_flaps'))
                self.problem.model.set_val('trajectory.I_landing_gear', self.data_problem.get_val('trajectory.I_landing_gear'))

            if self.settings['jet_mixing_source'] and not self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_problem.get_val('trajectory.jet_V'))
                self.problem.model.set_val('trajectory.jet_rho', self.data_problem.get_val('trajectory.jet_rho'))
                self.problem.model.set_val('trajectory.jet_A', self.data_problem.get_val('trajectory.jet_A'))
                self.problem.model.set_val('trajectory.jet_Tt', self.data_problem.get_val('trajectory.jet_Tt'))
            elif not self.settings['jet_mixing_source'] and self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_problem.get_val('trajectory.jet_V'))
                self.problem.model.set_val('trajectory.jet_A', self.data_problem.get_val('trajectory.jet_A'))
                self.problem.model.set_val('trajectory.jet_Tt', self.data_problem.get_val('trajectory.jet_Tt'))
                self.problem.model.set_val('trajectory.jet_M', self.data_problem.get_val('trajectory.jet_M'))
            elif self.settings['jet_mixing_source'] and self.settings['jet_shock_source']:
                self.problem.model.set_val('trajectory.jet_V', self.data_problem.get_val('trajectory.jet_V'))
                self.problem.model.set_val('trajectory.jet_rho', self.data_problem.get_val('trajectory.jet_rho'))
                self.problem.model.set_val('trajectory.jet_A', self.data_problem.get_val('trajectory.jet_A'))
                self.problem.model.set_val('trajectory.jet_Tt', self.data_problem.get_val('trajectory.jet_Tt'))
                self.problem.model.set_val('trajectory.jet_M', self.data_problem.get_val('trajectory.jet_M'))

            if self.settings['core_source']:
                if self.settings['core_turbine_attenuation_method'] == "ge":
                    self.problem.model.set_val('trajectory.core_mdot', self.data_problem.get_val('trajectory.core_mdot'))
                    self.problem.model.set_val('trajectory.core_Tt_i', self.data_problem.get_val('trajectory.core_Tt_i'))
                    self.problem.model.set_val('trajectory.core_Tt_j', self.data_problem.get_val('trajectory.core_Tt_j'))
                    self.problem.model.set_val('trajectory.core_Pt_i', self.data_problem.get_val('trajectory.core_Pt_i'))
                    self.problem.model.set_val('trajectory.turb_DTt_des', self.data_problem.get_val('trajectory.turb_DTt_des'))
                elif self.settings['core_turbine_attenuation_method'] == "pw":
                    self.problem.model.set_val('trajectory.core_mdot', self.data_problem.get_val('trajectory.core_mdot'))
                    self.problem.model.set_val('trajectory.core_Tt_i', self.data_problem.get_val('trajectory.core_Tt_i'))
                    self.problem.model.set_val('trajectory.core_Tt_j', self.data_problem.get_val('trajectory.core_Tt_j'))
                    self.problem.model.set_val('trajectory.core_Pt_i', self.data_problem.get_val('trajectory.core_Pt_i'))
                    self.problem.model.set_val('trajectory.turb_rho_i', self.data_problem.get_val('trajectory.turb_rho_i'))
                    self.problem.model.set_val('trajectory.turb_c_i', self.data_problem.get_val('trajectory.turb_c_i'))
                    self.problem.model.set_val('trajectory.turb_rho_e', self.data_problem.get_val('trajectory.turb_rho_e'))
                    self.problem.model.set_val('trajectory.turb_c_e', self.data_problem.get_val('trajectory.turb_c_e'))
                    
            if self.settings['fan_inlet_source'] or self.settings['fan_discharge_source']:
                self.problem.model.set_val('trajectory.fan_DTt', self.data_problem.get_val('trajectory.fan_DTt'))
                self.problem.model.set_val('trajectory.fan_mdot', self.data_problem.get_val('trajectory.fan_mdot'))
                self.problem.model.set_val('trajectory.fan_N', self.data_problem.get_val('trajectory.fan_N'))



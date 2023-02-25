import openmdao.api as om
from pyNA.src.trajectory_model.trajectory_data import TrajectoryData


class TimeHistory(om.Problem):
    
    def create(self, num_nodes, settings) -> None:
        
        self.model.add_subsystem('trajectory', TrajectoryData(num_nodes=num_nodes, settings=settings))

        return None

    def set_initial_conditions(self, settings, data) -> None:
        # Trajectory variables
        self.set_val('trajectory.t_s', data['t_s [s]'])
        self.set_val('trajectory.x', data['x [m]'])
        self.set_val('trajectory.y', data['y [m]'])
        self.set_val('trajectory.z', data['z [m]'])
        self.set_val('trajectory.v', data['v [m/s]'])
        self.set_val('trajectory.alpha', data['alpha [deg]'])
        self.set_val('trajectory.gamma', data['gamma [deg]'])
        self.set_val('trajectory.F_n', data['F_n [N]'])
        self.set_val('trajectory.tau', data['tau [-]'])
        self.set_val('trajectory.M_0', data['M_0 [-]'])
        self.set_val('trajectory.c_0', data['c_0 [m/s]'])
        self.set_val('trajectory.T_0', data['T_0 [K]'])
        self.set_val('trajectory.p_0', data['p_0 [Pa]'])
        self.set_val('trajectory.rho_0', data['rho_0 [kg/m3]'])
        self.set_val('trajectory.mu_0', data['mu_0 [kg/ms]'])
        self.set_val('trajectory.I_0', data['I_0 [kg/m2s]'])
        
        if settings['airframe_source']:
            self.set_val('trajectory.theta_flaps', data['theta_flaps [deg]'])
            self.set_val('trajectory.I_lg', data['I_lg [-]'])

        # Engine variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            self.set_val('trajectory.jet_V', data['jet_V [m/s]'])
            self.set_val('trajectory.jet_rho', data['jet_rho [kg/m3]'])
            self.set_val('trajectory.jet_A', data['jet_A [m2]'])
            self.set_val('trajectory.jet_Tt', data['jet_Tt [K]'])
        elif not settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.set_val('trajectory.jet_V', data['jet_V [m/s]'])
            self.set_val('trajectory.jet_A', data['jet_A [m2]'])
            self.set_val('trajectory.jet_Tt', data['jet_Tt [K]'])
            self.set_val('trajectory.jet_M', data['jet_M [-]'])
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.set_val('trajectory.jet_V', data['jet_V [m/s]'])
            self.set_val('trajectory.jet_rho', data['jet_rho [kg/m3]'])
            self.set_val('trajectory.jet_A', data['jet_A [m2]'])
            self.set_val('trajectory.jet_Tt', data['jet_Tt [K]'])
            self.set_val('trajectory.jet_M', data['jet_M [-]'])

        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == "ge":
                self.set_val('trajectory.core_mdot', data['core_mdot [kg/s]'])
                self.set_val('trajectory.core_Tt_i', data['core_Tt_i [K]'])
                self.set_val('trajectory.core_Tt_j', data['core_Tt_j [K]'])
                self.set_val('trajectory.core_Pt_i', data['core_Pt_i [Pa]'])
                self.set_val('trajectory.turb_DTt_des', data['turb_DTt_des [K]'])
            elif settings['core_turbine_attenuation_method'] == "pw":
                self.set_val('trajectory.core_mdot', data['core_mdot [kg/s]'])
                self.set_val('trajectory.core_Tt_i', data['core_Tt_i [K]'])
                self.set_val('trajectory.core_Tt_j', data['core_Tt_j [K]'])
                self.set_val('trajectory.core_Pt_i', data['core_Pt_i [Pa]'])
                self.set_val('trajectory.turb_rho_i', data['turb_rho_i [kg/m3]'])
                self.set_val('trajectory.turb_c_i', data['turb_c_i [m/s]'])
                self.set_val('trajectory.turb_rho_e', data['turb_rho_e [kg/m3]'])
                self.set_val('trajectory.turb_c_e', data['turb_c_e [m/s]'])
                
        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            self.set_val('trajectory.fan_DTt', data['fan_DTt [K]'])
            self.set_val('trajectory.fan_mdot', data['fan_mdot [kg/s]'])
            self.set_val('trajectory.fan_N', data['fan_N [rpm]'])
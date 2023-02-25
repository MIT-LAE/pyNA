import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openmdao.api as om
import dymos as dm
import os
import pdb

from pyNA.src.trajectory_model.trajectory_data import TrajectoryData
from pyNA.src.trajectory_model.take_off_phase_ode import TakeOffPhaseODE
from pyNA.src.trajectory_model.mux import Mux

from pyNA.src.trajectory_model.phases.groundroll import GroundRoll
from pyNA.src.trajectory_model.phases.rotation import Rotation
from pyNA.src.trajectory_model.phases.liftoff import LiftOff
from pyNA.src.trajectory_model.phases.vnrs import Vnrs
from pyNA.src.trajectory_model.phases.cutback import CutBack


class Trajectory:
    
    def __init__(self, settings, mode:str) -> None:
        
        """
        
        :param mode: use 'data' to input the time history of the trajectory variables and engine operational variables using .csv files; use 'compute' to model the trajectory and engine using Dymos
        :type mode: str

        """

        self.mode = mode
        if self.mode == 'model':
            self.path = om.Problem()

        elif self.mode == 'data':
            # Load data .csv files
            self.time_history = pd.DataFrame()
            Trajectory.load_time_history_csv(self, settings=settings)

            # Create openmdao problem
            self.path = om.Problem()
            self.path.model.add_subsystem('trajectory', TrajectoryData(num_nodes=self.n_t, settings=settings))

    def load_time_history_csv(self, settings) -> None:
        
        """
        Load engine and trajectory time history from .csv file.

        :param settings:
        :type settings:

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        try:
            self.time_history = pd.read_csv(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/' + settings['time_history_file_name'])
        except:
            raise ValueError(settings['time_history_file_name'] + ' file not found at ' + settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/')
        
        # Compute number of time steps in the time history
        self.n_t = np.size(self.time_history['t_s [s]'])

        return None

    def set_time_history_initial_conditions(self, settings):
        # Trajectory variables
        self.path.set_val('trajectory.t_s', self.time_history['t_s [s]'])
        self.path.set_val('trajectory.x', self.time_history['x [m]'])
        self.path.set_val('trajectory.y', self.time_history['y [m]'])
        self.path.set_val('trajectory.z', self.time_history['z [m]'])
        self.path.set_val('trajectory.v', self.time_history['v [m/s]'])
        self.path.set_val('trajectory.alpha', self.time_history['alpha [deg]'])
        self.path.set_val('trajectory.gamma', self.time_history['gamma [deg]'])
        self.path.set_val('trajectory.F_n', self.time_history['F_n [N]'])
        self.path.set_val('trajectory.tau', self.time_history['tau [-]'])
        self.path.set_val('trajectory.M_0', self.time_history['M_0 [-]'])
        self.path.set_val('trajectory.c_0', self.time_history['c_0 [m/s]'])
        self.path.set_val('trajectory.T_0', self.time_history['T_0 [K]'])
        self.path.set_val('trajectory.p_0', self.time_history['p_0 [Pa]'])
        self.path.set_val('trajectory.rho_0', self.time_history['rho_0 [kg/m3]'])
        self.path.set_val('trajectory.mu_0', self.time_history['mu_0 [kg/ms]'])
        self.path.set_val('trajectory.I_0', self.time_history['I_0 [kg/m2s]'])
        
        if settings['airframe_source']:
            self.path.set_val('trajectory.theta_flaps', self.time_history['theta_flaps [deg]'])
            self.path.set_val('trajectory.I_lg', self.time_history['I_lg [-]'])

        # Engine variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            self.path.set_val('trajectory.jet_V', self.time_history['jet_V [m/s]'])
            self.path.set_val('trajectory.jet_rho', self.time_history['jet_rho [kg/m3]'])
            self.path.set_val('trajectory.jet_A', self.time_history['jet_A [m2]'])
            self.path.set_val('trajectory.jet_Tt', self.time_history['jet_Tt [K]'])
        elif not settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.path.set_val('trajectory.jet_V', self.time_history['jet_V [m/s]'])
            self.path.set_val('trajectory.jet_A', self.time_history['jet_A [m2]'])
            self.path.set_val('trajectory.jet_Tt', self.time_history['jet_Tt [K]'])
            self.path.set_val('trajectory.jet_M', self.time_history['jet_M [-]'])
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.path.set_val('trajectory.jet_V', self.time_history['jet_V [m/s]'])
            self.path.set_val('trajectory.jet_rho', self.time_history['jet_rho [kg/m3]'])
            self.path.set_val('trajectory.jet_A', self.time_history['jet_A [m2]'])
            self.path.set_val('trajectory.jet_Tt', self.time_history['jet_Tt [K]'])
            self.path.set_val('trajectory.jet_M', self.time_history['jet_M [-]'])

        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == "ge":
                self.path.set_val('trajectory.core_mdot', self.time_history['core_mdot [kg/s]'])
                self.path.set_val('trajectory.core_Tt_i', self.time_history['core_Tt_i [K]'])
                self.path.set_val('trajectory.core_Tt_j', self.time_history['core_Tt_j [K]'])
                self.path.set_val('trajectory.core_Pt_i', self.time_history['core_Pt_i [Pa]'])
                self.path.set_val('trajectory.turb_DTt_des', self.time_history['turb_DTt_des [K]'])
            elif settings['core_turbine_attenuation_method'] == "pw":
                self.path.set_val('trajectory.core_mdot', self.time_history['core_mdot [kg/s]'])
                self.path.set_val('trajectory.core_Tt_i', self.time_history['core_Tt_i [K]'])
                self.path.set_val('trajectory.core_Tt_j', self.time_history['core_Tt_j [K]'])
                self.path.set_val('trajectory.core_Pt_i', self.time_history['core_Pt_i [Pa]'])
                self.path.set_val('trajectory.turb_rho_i', self.time_history['turb_rho_i [kg/m3]'])
                self.path.set_val('trajectory.turb_c_i', self.time_history['turb_c_i [m/s]'])
                self.path.set_val('trajectory.turb_rho_e', self.time_history['turb_rho_e [kg/m3]'])
                self.path.set_val('trajectory.turb_c_e', self.time_history['turb_c_e [m/s]'])
                
        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            self.path.set_val('trajectory.fan_DTt', self.time_history['fan_DTt [K]'])
            self.path.set_val('trajectory.fan_mdot', self.time_history['fan_mdot [kg/s]'])
            self.path.set_val('trajectory.fan_N', self.time_history['fan_N [rpm]'])

    def create_model(self, settings, airframe, engine, trajectory_mode='cutback', objective='time'):
        
        self.traj = dm.Trajectory()
        self.path.add_subsystem('phases', self.traj)

        # Create the trajectory phases 
        for phase_name in self.phase_name_lst:
            opts = {'phase': phase_name, 'settings': settings, 'airframe': airframe, 'engine': engine, 'objective': objective}
            
            if phase_name == 'groundroll':
                self.groundroll = GroundRoll(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.groundroll.create(settings, airframe, engine, objective)
                self.traj.add_phase(phase_name, self.groundroll)
            
            elif phase_name == 'rotation':
                self.rotation = Rotation(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.rotation.create(settings, airframe, engine, objective)
                self.traj.add_phase(phase_name, self.rotation)

            elif phase_name == 'liftoff':
                self.liftoff = LiftOff(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.liftoff.create(settings, airframe, engine, objective)
                self.traj.add_phase(phase_name, self.liftoff)

            elif phase_name == 'vnrs':
                self.vnrs = Vnrs(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.vnrs.create(settings, airframe, engine, objective, trajectory_mode)
                self.traj.add_phase(phase_name, self.vnrs)
                
            elif phase_name == 'cutback':
                self.cutback = CutBack(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.cutback.create(settings, airframe, engine, objective, trajectory_mode)
                self.traj.add_phase(phase_name, self.cutback)        

        # Link phases
        if 'rotation' in self.phase_name_lst:
            self.traj.link_phases(phases=['groundroll', 'rotation'], vars=['time', 'x', 'v', 'alpha'])    
            if objective == 'noise' and settings['phld']:
                self.traj.add_linkage_constraint(phase_a='groundroll', phase_b='rotation', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'liftoff' in self.phase_name_lst:
            self.traj.link_phases(phases=['rotation', 'liftoff'], vars=['time', 'x', 'z', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['phld']:
                self.traj.add_linkage_constraint(phase_a='rotation', phase_b='liftoff', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'vnrs' in self.phase_name_lst:
            self.traj.link_phases(phases=['liftoff', 'vnrs'],  vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['ptcb']:
                self.traj.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='TS', var_b='TS', loc_a='final', loc_b='initial')

        if 'cutback' in self.phase_name_lst:
            if trajectory_mode == 'flyover':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'z', 'v', 'alpha', 'gamma'])
            elif trajectory_mode == 'cutback':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['ptcb']:
                self.traj.add_linkage_constraint(phase_a='vnrs', phase_b='cutback', var_a='TS', var_b='TS', loc_a='final', loc_b='initial')

        # Mux trajectory and engine variables
        Trajectory.get_mux_input_output_size(self)
        mux_t = self.model.add_subsystem(name='trajectory', subsys=Mux(objective=objective, input_size_array=self.mux_input_size_array, output_size=self.mux_output_size, case_name=self.case_name, output_directory_name=self.output_directory_name))
        trajectory_var = Trajectory.get_trajectory_var_lst(atmosphere_type)

        for var in trajectory_var.keys():
            
            if var == 'time':
                mux_t.add_var('t_s', units=trajectory_var[var])
            else:
                mux_t.add_var(var, units=trajectory_var[var])

            for j, phase_name in enumerate(self.phase_name_lst):
                if var == 'time':
                    self.model.connect('phases.' + phase_name + '.interpolated.' + var, 'trajectory.t_s_' + str(j))

                elif var in ['x', 'v']:
                    self.model.connect('phases.' + phase_name + '.interpolated.states:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['z', 'gamma']:
                    if phase_name in {'groundroll','rotation'}:
                        self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                    else:
                        self.model.connect('phases.' + phase_name + '.interpolated.states:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['alpha']:
                    if phase_name in {'groundroll'}:
                        self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var,'trajectory.' + var + '_' + str(j))
                    # elif phase_name in {'rotation', 'liftoff'}:
                    elif phase_name in {'rotation'}:
                        self.model.connect('phases.' + phase_name + '.interpolated.states:' + var,'trajectory.' + var + '_' + str(j))
                    else:
                        self.model.connect('phases.' + phase_name + '.interpolated.controls:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['TS']:
                    if objective == 'noise' and ptcb:
                        if phase_name in ['groundroll', 'rotation', 'liftoff']:
                            self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'vnrs':
                            self.model.connect('phases.' + phase_name + '.interpolated.controls:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'cutback':
                            self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                    else:
                        if phase_name in ['groundroll', 'rotation', 'liftoff', 'vnrs']:
                            self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'cutback':
                            self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['theta_flaps', 'theta_slats', 'y', 'I_landing_gear']:
                    self.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['L', 'D', 'eas', 'n','M_0', 'p_0','rho_0', 'T_0', 'c_0', 'c_bar', 'mu_0', 'I_0', 'mdot_NOx', 'EINOx', 'c_l', 'c_d', 'c_l_max']:
                    self.model.connect('phases.' + phase_name + '.interpolated.' + var, 'trajectory.' + var + '_' + str(j))

        mux_e = self.model.add_subsystem(name='engine', subsys=Mux(objective=objective, input_size_array=self.mux_input_size_array, output_size=self.mux_output_size, case_name=self.case_name, output_directory_name=self.output_directory_name))
        for var in engine.deck_variables.keys():
            
            mux_e.add_var(var, units=engine.deck_variables[var])
            
            for j, phase_name in enumerate(self.phase_name_lst):
                self.model.connect('phases.' + phase_name + '.interpolated.' + var, 'engine.' + var + '_' + str(j))

        return 

    def set_model_initial_conditions():
        pass

    def compute_path(self, settings, objective='time'):
        if self.mode == 'model':
            print(objective)
            pass

        elif self.mode == 'data':
            self.path.setup()
            Trajectory.set_time_history_initial_conditions(self, settings)
            self.path.run_model()
  
    def check_convergence(self, filename: str) -> bool:
        """
        Checks convergence of case using optimizer output file.

        :param settings: pyna settings
        :type settings: Settings
        :param filename: file name of IPOPT output
        :type filename: str

        :return: converged
        :rtype: bool
        """

        # Save convergence info for trajectory
        # Read IPOPT file
        file_ipopt = open(self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + filename, 'r')
        ipopt = file_ipopt.readlines()
        file_ipopt.close()

        # Check if convergence summary excel file exists
        cnvg_file_name = self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + 'Convergence.csv'
        if not os.path.isfile(cnvg_file_name):
            file_cvg = open(cnvg_file_name, 'w')
            file_cvg.writelines("Trajectory name , Execution date/time,  Converged")
        else:
            file_cvg = open(cnvg_file_name, 'a')

        # Write convergence output to file
        # file = open(cnvg_file_name, 'a')
        if ipopt[-1] in {'EXIT: Optimal Solution Found.\n', 'EXIT: Solved To Acceptable Level.\n'}:
            file_cvg.writelines("\n" + self.output_file_name + ", " + str(dt.datetime.now()) + ", Converged")
            converged = True
        else:
            file_cvg.writelines("\n" + self.output_file_name + ", " + str(dt.datetime.now()) + ", Not converged")
            converged = False
        file_cvg.close()

        return converged

    def plot_ipopt_convergence_data():
        pass

    def plot_path(self):

        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use('plot.mplstyle')

        ax[0,0].plot(self.path.get_val('trajectory.x'), self.path.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.path.get_val('trajectory.t_s'), self.path.get_val('trajectory.v'), '-', color='k')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.path.get_val('trajectory.t_s'), self.path.get_val('trajectory.gamma'), '-', color='k')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.path.get_val('trajectory.t_s'), 1 / 1000. * self.path.get_val('trajectory.F_n'), '-', color='k')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.path.get_val('trajectory.t_s'), self.path.get_val('trajectory.tau'), '-', color='k')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$TS$ [-]')
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.path.get_val('trajectory.t_s'), self.path.get_val('trajectory.alpha'), '-', color='k')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None



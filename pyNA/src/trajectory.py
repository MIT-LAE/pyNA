import pdb
import os
import pandas as pd
import dymos as dm
import numpy as np
import datetime as dt
from typing import Union
import openmdao.api as om
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_src.atmosphere import Atmosphere
from pyNA.src.engine import Engine
from scipy.interpolate import RegularGridInterpolator
from pyNA.src.trajectory_src.trajectory_ode import TrajectoryODE
from pyNA.src.trajectory_src.mux import Mux


class Trajectory:
    """
    The trajectory module contains the methods to compute the take-off trajectory used by pyNA.
    """

    def __init__(self, n_order):

        # Initialize path
        self.path = pd.DataFrame
        self.n_t = np.int64

        # Initialize phases
        self.phase_name_lst = ['groundroll', 'rotation', 'liftoff', 'vnrs', 'cutback']
        self.phases = dict()

        # Compute transcription for the phases
        self.num_segments = []
        self.transcription_order = []
        self.transcription_phases = []
        self.phase_size = []
        for i, phase in enumerate(self.phase_name_lst):
            if phase == 'groundroll':
                self.num_segments.append(3)
                self.transcription_order.append(3)
            elif phase == 'rotation':
                self.num_segments.append(3)
                self.transcription_order.append(3)
            elif phase == 'liftoff':
                self.num_segments.append(4)
                self.transcription_order.append(3)
            elif phase == 'vnrs':
                self.num_segments.append(7)
                self.transcription_order.append(3)
            elif phase == 'cutback':
                self.num_segments.append(12)
                self.transcription_order.append(3)
            
            self.transcription_phases.append(dm.GaussLobatto(num_segments=self.num_segments[i], order=self.transcription_order[i], compressed=True, solve_segments=False))
            self.transcription_phases[i].init_grid()
            self.phase_size.append(self.num_segments[i] * self.transcription_order[i]+1)

        # Compute size of the muxed trajectory
        self.trajectory_size = Trajectory.compute_size_output_mux(size_inputs=self.phase_size)

        return

    @staticmethod
    def get_engine_variables(settings: Settings) -> Union[list, list]:
        """
        Get the engine parameters to compute during the trajectory computations.

        :param settings: pyna settings
        :type settings: Settings

        :return: (engine_var, engine_var_units)
        :rtype: (list, list)
        """
        # Engine variables
        engine_var = ['W_f', 'Tti_c', 'Pti_c']
        engine_var_units = ['kg/s', 'K', 'Pa']

        # Jet parameters
        engine_var.extend(['V_j', 'rho_j', 'A_j', 'Tt_j', 'M_j'])
        engine_var_units.extend(['m/s', 'kg/m**3', 'm**2', 'K', None])

        # Core parameters
        if settings.method_core_turb == 'GE':
            engine_var.extend(['mdoti_c', 'Ttj_c', 'DTt_des_c'])
            engine_var_units.extend(['kg/s', 'K', 'K'])
        elif settings.method_core_turb == 'PW':
            engine_var.extend(['mdoti_c', 'Ttj_c', 'DTt_des_c', 'rho_te_c', 'c_te_c', 'rho_ti_c', 'c_ti_c'])
            engine_var_units.extend(['kg/s', 'K', 'K', 'kg/m**3', 'm/s', 'kg/m**3', 'm/s'])
        
        # Fan parameters
        engine_var.extend(['DTt_f', 'mdot_f', 'N_f', 'A_f', 'd_f'])
        engine_var_units.extend(['K', 'kg/s', 'rpm', 'm**2', 'm'])

        return engine_var, engine_var_units

    def compute_size_output_mux(size_inputs: np.ndarray):
        """
        Compute vector size of the muxed trajectory.

        :param size_inputs: 
        :type size_inputs: np.ndarray 

        """

        mux_num = len(size_inputs)
        
        size_output = 0
        for i in range(mux_num):
            
            # Add input size to output vector
            if i < mux_num-1:
                size_output = size_output + (size_inputs[i]-1)
            else:
                size_output = size_output + (size_inputs[i])
  
        return size_output

    # Compute minimum thrust requirement
    def compute_minimum_TS(settings: Settings, ac: Aircraft, engine: Engine, z_lst=[1300, 1300], v_lst=[250, 250], gamma_lst=[0.0, np.arctan(0.04)*180/np.pi]):
        
        # Initialize limiting cases
        case_lst = ['OEI', '4%CG']
        nr_engine_lst = np.array([ac.n_eng - 1, ac.n_eng])

        # Create engine deck interpolant
        F_n_interp = RegularGridInterpolator((engine.deck['z'], engine.deck['M_0'], engine.deck['TS']), engine.deck['F_n'])
        
        sol = dict()
        sol['alpha'] = np.zeros(2)
        sol['c_l'] = np.zeros(2)
        sol['c_d'] = np.zeros(2) 
        sol['F_avail'] = np.zeros(2)
        sol['F_req'] = np.zeros(2)
        sol['TS'] = np.zeros(2)
        
        for i, case in enumerate(case_lst):
            # Compute atmospheric properties at ac.z_max
            prob_atm = om.Problem()
            prob_atm.model.add_subsystem("atm", Atmosphere(num_nodes=1, settings=settings))
            prob_atm.setup(force_alloc_complex=True)
            prob_atm.set_val('atm.z', z_lst[i])
            prob_atm.run_model()
            rho_0 = prob_atm.get_val('atm.rho_0')
            c_0 = prob_atm.get_val('atm.c_0')
            
            # Lift requirement for steady flight
            L = 9.80665 * ac.mtow * np.cos(gamma_lst[i] * np.pi / 180.)
            sol['c_l'][i] = L / (0.5* rho_0 * v_lst[i] ** 2 * ac.af_S_w)
            
            settings.theta_flaps = 10.
            if settings.ac_name == 'stca':
                settings.theta_slats = -6.
            elif settings.ac_name == 'a10':
                settings.theta_slats = 0.
            
            c_l_interp = RegularGridInterpolator((ac.aero['alpha'], ac.aero['theta_flaps'], ac.aero['theta_slats']), ac.aero['c_l'])        
            c_l_data = c_l_interp((ac.aero['alpha'], settings.theta_flaps, settings.theta_slats))
            
            c_d_interp = RegularGridInterpolator((ac.aero['alpha'], ac.aero['theta_flaps'], ac.aero['theta_slats']), ac.aero['c_d'])
            c_d_data = c_d_interp((ac.aero['alpha'], settings.theta_flaps, settings.theta_slats))
            
            # Before stall
            if sol['c_l'][i] <= np.max(c_l_data):         
                # Compute required angle of attack to meet lift coefficient
                sol['alpha'][i] = np.interp(sol['c_l'][i], c_l_data, ac.aero['alpha'])

                # Compute corresponding drag coefficient
                sol['c_d'][i] = np.interp(sol['alpha'][i], ac.aero['alpha'], c_d_data)

            else:
                sol['alpha'][i] = 100.
                sol['c_d'][i] = 100.
                
            # Compute aircraft total thrust requirement
            T = (sol['c_d'][i] * 0.5 * rho_0 * v_lst[i] ** 2 * ac.af_S_w) + ac.mtow * 9.80065 * np.sin(gamma_lst[i] * np.pi / 180.)

            # Compute thrust requirement per engine
            sol['F_req'][i] = T / nr_engine_lst[i]
            
            # Compute thrust available
            sol['F_avail'][i] = F_n_interp((z_lst[i], v_lst[i] / c_0, 1.))[0]

            # Compute minimum thrust setting
            sol['TS'][i] = sol['F_req'][i] / sol['F_avail'][i]
                                
        return sol

    # @staticmethod
    # def compute_minimum_TS(settings: Settings, ac: Aircraft, engine: Engine) -> np.float64:
    #     """
    #     Compute minimum cutback thrust-setting meeting the 4%CG and one-engine-inoperative (OEI) airworthiness requirements.
        
    #     :param settings: pyNA settings
    #     :type settings: Settings
    #     :param ac: aircraft parameters
    #     :type ac: Aircraft
    #     :param engine: engine parameters
    #     :param engine: Engine
    #     :return: TS_max
    #     :rtype: np.float64

    #     """

    #     # Initialize limiting cases
    #     gamma_lst = np.array([0, 2.3])
    #     nr_engine_lst = np.array([ac.n_eng - 1, ac.n_eng])
        
    #     alpha = np.zeros(2)
    #     TS_lst = np.zeros(2)

    #     for cc, case in enumerate(['OEI', '4%CG']):
    #         # Compute atmospheric properties at ac.z_max
    #         prob_atm = om.Problem()
    #         prob_atm.model.add_subsystem("atm", Atmosphere(num_nodes=1, settings=settings))
    #         prob_atm.setup(force_alloc_complex=True)
    #         prob_atm.set_val('atm.z', ac.z_max)
    #         prob_atm.run_model()
    #         rho_0 = prob_atm.get_val('atm.rho_0')
    #         c_0 = prob_atm.get_val('atm.c_0')

    #         # Lift requirement for horizontal, steady climbing flight
    #         L = 9.80665 * ac.mtow * np.cos(gamma_lst[cc] * np.pi / 180.)
    #         c_l = L / (0.5* rho_0 * ac.v_max ** 2 * ac.af_S_w)

    #         # Compute required angle of attack to meet lift coefficient
    #         c_l_interp = RegularGridInterpolator((ac.aero['alpha'], ac.aero['theta_flaps'], ac.aero['theta_slats']), ac.aero['c_l'])
    #         c_l_data = c_l_interp((ac.aero['alpha'], settings.theta_flaps, settings.theta_slats))
    #         alpha[cc] = np.interp(c_l, c_l_data, ac.aero['alpha'])

    #         # Compute corresponding drag coefficient
    #         c_d_interp = RegularGridInterpolator((ac.aero['alpha'], ac.aero['theta_flaps'], ac.aero['theta_slats']), ac.aero['c_d'])
    #         c_d_data = c_d_interp((ac.aero['alpha'], settings.theta_flaps, settings.theta_slats))
    #         c_d = np.interp(alpha[cc], ac.aero['alpha'], c_d_data)
            
    #         # Compute thrust requirement
    #         D = (c_d * 0.5 * rho_0 * ac.v_max ** 2 * ac.af_S_w) + ac.mtow * 9.80065 * np.sin(gamma_lst[cc] * np.pi / 180.)
    #         F_req = D / nr_engine_lst[cc]

    #         # Compute thrust available
    #         F_n_interp = RegularGridInterpolator((engine.deck['z'], engine.deck['M_0'], engine.deck['TS']), engine.deck['F_n'])
    #         F_avl = F_n_interp([ac.z_max, ac.v_max / c_0, 1.])[0]

    #         # Compute minimum thrust setting
    #         TS_lst[cc] = F_req / F_avl
    #         # Print results
    #         print(case, 'engine thrust-setting requirement: ', np.round(TS_lst[cc], 3))

    #     # Compute TS_max
    #     TS_max = max(TS_lst)

    #     return TS_max

    def load_time_series(self, settings: Settings) -> None:
        """
        Loads predefined trajectory timeseries.

        :param settings: pyna settings
        :type settings: Settings

        :return: None

        """

        # Load trajectory data for the specific observer
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.path = pd.read_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/trajectory/' + settings.output_directory_name + '/' + settings.trajectory_file_name)
        self.n_t = np.size(self.path['t_source [s]'])

        return None

    def load_operating_point(self, settings:Settings, time_step: int) -> None:
        """
        Loads predefined trajectory timeseries.

        :param settings: pyna settings
        :type settings: Settings
        :param time_step: time step of the operating point in the trajectory time series
        :type time_step: int

        :return: None

        """

        # Load trajectory data for the specific observer
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.path = pd.read_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/trajectory/' + settings.output_directory_name + '/' + settings.trajectory_file_name)
        
        # Select operating point
        cols = self.path.columns
        op_point = pd.DataFrame(np.reshape(self.path.values[time_step, :], (1, len(cols))))
        op_point.columns = cols

        # Duplicate operating for theta range (np.linspace(0, 180, 19))
        self.path = pd.DataFrame()
        for i in np.arange(19):
            self.path = self.path.append(op_point)

        self.n_t = 19

        return None  

    def setup(self, problem: om.Problem, settings: Settings, ac: Aircraft, engine: Engine, trajectory_mode: str, objective: str) -> None:
        """
        Setup take-off trajectory module using the following phases:

        * ``Ground roll``:  acceleration from V=0
        * ``Flaps down``:   accelerate further to V=kVstall while deploying flaps during x seconds
        * ``Rotation``:     rotate at dalpha/dt = cnst until load factor n=1;
        * ``Lift off``:     climb until obstacle is cleared (35ft).
        * ``VNRS``:         any VNRS can be applied in this phase, e.g. PTCB or PHLD
        * ``cutback``:      a pilot-initiated thrust cut-back to a constant thrust-setting is applied

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
        :param engine: engine parameters
        :param engine: Engine
        :param objective: optimization objective
        :type objective: str

        :return: None

        """

        # Set solver settings for the problem
        problem.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        problem.driver.opt_settings['print_level'] = 5
        problem.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

        problem.driver.declare_coloring(tol=1e-12)
        problem.model.linear_solver = om.LinearRunOnce()
        problem.driver.opt_settings['output_file'] = settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_directory_name + '/IPOPT_trajectory_convergence.out'

        if objective == 'noise':
            problem.driver.opt_settings['tol'] = 1e-2
            problem.driver.opt_settings['acceptable_tol'] = 1e-1
        else:
            problem.driver.opt_settings['tol'] = settings.tol
            problem.driver.opt_settings['acceptable_tol'] = 1e-2

        problem.driver.opt_settings['max_iter'] = settings.max_iter
        problem.driver.opt_settings['mu_strategy'] = 'adaptive'
        problem.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        problem.driver.opt_settings['mu_init'] = 0.01
        problem.driver.opt_settings['constr_viol_tol'] = 1e-3
        problem.driver.opt_settings['compl_inf_tol'] = 1e-3
        problem.driver.opt_settings['acceptable_iter'] = 0
        problem.driver.opt_settings['acceptable_constr_viol_tol'] = 1e-1
        problem.driver.opt_settings['acceptable_compl_inf_tol'] = 1e-1
        problem.driver.opt_settings['acceptable_obj_change_tol'] = 1e-1

        # Setup trajectory and initialize trajectory transcription and compute number of points per phase
        traj = dm.Trajectory()
        problem.model.add_subsystem('phases', traj)

        # Get engine variables
        engine_var, engine_var_units = Trajectory.get_engine_variables(settings)

        # Compute the minimum thrust-setting based on 4% climb gradient and OEI requirement
        if settings.TS_cutback:
            TS_min = settings.TS_cutback
        else:
            sol = Trajectory.compute_minimum_TS(settings, ac, engine, z_lst=[1300*0.3048, 1300*0.3048], v_lst=[ac.v_max, ac.v_max], gamma_lst=[0.0, np.arctan(0.04)*180/np.pi])
            TS_min = np.max(sol['TS'])

        # Phase 1: ground roll
        if 'groundroll' in self.phase_name_lst:
            opts = {'phase': 'groundroll', 'ac': ac, 'engine': engine, 'settings': settings, 'objective': objective}
            self.phases['groundroll'] = dm.Phase(ode_class=TrajectoryODE, ode_init_kwargs=opts, transcription=self.transcription_phases[0])
            self.phases['groundroll'].set_time_options(fix_initial=True, duration_bounds=(0, 60), duration_ref=100.)
            self.phases['groundroll'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=True, fix_final=False, ref=1000.)
            self.phases['groundroll'].add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=True, fix_final=False, ref=100.)
            self.phases['groundroll'].add_state('alpha', targets='alpha', rate_source='flight_dynamics.alpha_dot', units='deg', fix_initial=True, fix_final=False, lower=ac.aero['alpha'][0], upper=ac.aero['alpha'][-1], ref=1.)
            self.phases['groundroll'].add_parameter('z', targets='z', units='m', val=0., dynamic=True,include_timeseries=True)
            self.phases['groundroll'].add_parameter('gamma', targets='gamma', units='deg', val=0., dynamic=True, include_timeseries=True)
            self.phases['groundroll'].add_parameter('TS', targets='propulsion.TS', units=None, val=settings.TS_to, dynamic=True, include_timeseries=True)
            self.phases['groundroll'].add_parameter('TS_min', units=None, val=1, dynamic=True, include_timeseries=True)
            if settings.PKROT:
                self.phases['groundroll'].add_parameter('k_rot', targets='flight_dynamics.k_rot', units=None, lower=1.1, upper=1.6, dynamic=False, val=ac.k_rot, opt=True)
            else:
                self.phases['groundroll'].add_parameter('k_rot', targets='flight_dynamics.k_rot', units=None, dynamic=False, val=ac.k_rot, opt=False)
            # PHLD
            # if objective == 'noise' and settings.PHLD:
                # self.phases['groundroll'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=ac.aero['theta_flaps_c_d_min_gr'], dynamic=True, include_timeseries=True, opt=False)
            # else:
            self.phases['groundroll'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, dynamic=True, include_timeseries=True)
            self.phases['groundroll'].add_parameter('theta_slats', targets='theta_slats', units='deg', val=settings.theta_slats, dynamic=True, include_timeseries=True)
            self.phases['groundroll'].add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_size[0]-1,order=3, solve_segments=False, compressed=True), subset='state_input')
            self.phases['groundroll'].add_boundary_constraint('flight_dynamics.v_rot_residual', equals=0., loc='final', ref=100, units='m/s')

        # Phase 2: rotation phase
        if 'rotation' in self.phase_name_lst:
            opts = {'phase': 'rotation', 'ac': ac, 'engine': engine, 'settings': settings, 'objective': objective}
            self.phases['rotation'] = dm.Phase(ode_class=TrajectoryODE, transcription=self.transcription_phases[1], ode_init_kwargs=opts)
            self.phases['rotation'].set_time_options(initial_bounds=(20, 60), duration_bounds=(0, 60), initial_ref=100., duration_ref=100.)
            self.phases['rotation'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=1000.)
            self.phases['rotation'].add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
            self.phases['rotation'].add_state('alpha', targets='alpha', rate_source='flight_dynamics.alpha_dot', units='deg', fix_initial=False, fix_final=False, lower=ac.aero['alpha'][0], upper=ac.aero['alpha'][-1], ref=10.)
            self.phases['rotation'].add_parameter('z', targets='z', units='m', val=0., dynamic=True,include_timeseries=True)
            self.phases['rotation'].add_parameter('gamma', targets='gamma', units='deg', val=0., dynamic=True, include_timeseries=True)
            self.phases['rotation'].add_parameter('TS', targets='propulsion.TS', units=None, val=settings.TS_to, dynamic=True, include_timeseries=True)
            self.phases['rotation'].add_parameter('TS_min', units=None, val=1, dynamic=True, include_timeseries=True)
            # PHLD
            # if objective == 'noise' and settings.PHLD:
                # self.phases['rotation'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, lower=ac.aero['theta_flaps'][0], upper=ac.aero['theta_flaps'][-1], dynamic=True, include_timeseries=True, opt=True, ref=10.)
                # self.phases['rotation'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=15., dynamic=True, include_timeseries=True)
            # else:
            self.phases['rotation'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, dynamic=True, include_timeseries=True)
            self.phases['rotation'].add_parameter('theta_slats', targets='theta_slats', units='deg', val=settings.theta_slats, dynamic=True, include_timeseries=True)
            self.phases['rotation'].add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_size[1]-1, order=3, solve_segments=False, compressed=True), subset='state_input')
            self.phases['rotation'].add_boundary_constraint('flight_dynamics.n', equals=1., loc='final', ref=1, units=None)

        # Phase 3: lift-off phase
        if 'liftoff' in self.phase_name_lst:
            opts = {'phase': 'liftoff', 'ac': ac, 'engine': engine, 'settings': settings, 'objective': objective}
            self.phases['liftoff'] = dm.Phase(ode_class=TrajectoryODE, transcription=self.transcription_phases[2], ode_init_kwargs=opts)
            self.phases['liftoff'].set_time_options(initial_bounds=(20, 150), duration_bounds=(0, 500), initial_ref=100., duration_ref=100., fix_duration=False)
            self.phases['liftoff'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=10000.)
            self.phases['liftoff'].add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=False, fix_final=True, ref=10.)
            self.phases['liftoff'].add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
            self.phases['liftoff'].add_state('gamma', rate_source='flight_dynamics.gamma_dot', units='deg', fix_initial=False, fix_final=False, ref=10.)
            self.phases['liftoff'].add_control('alpha', targets='alpha', units='deg', lower=ac.aero['alpha'][0], upper=ac.aero['alpha'][-1], rate_continuity=True, rate_continuity_scaler=1.0, rate2_continuity=False, opt=True, ref=10.)
            self.phases['liftoff'].add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_size[2]-1, order=3, solve_segments=False, compressed=True), subset='state_input')
            self.phases['liftoff'].add_path_constraint(name='flight_dynamics.gamma_dot', lower=0., units='deg/s')
            self.phases['liftoff'].add_path_constraint(name='flight_dynamics.v_dot', lower=0., units='m/s**2')
            self.phases['liftoff'].add_parameter('TS', targets='propulsion.TS', units=None, val=settings.TS_to, dynamic=True, include_timeseries=True)
            self.phases['liftoff'].add_parameter('TS_min', units=None, val=1, dynamic=True, include_timeseries=True)
            # PHLD
            # if objective == 'noise' and settings.PHLD:
                # self.phases['liftoff'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, lower=ac.aero['theta_flaps'][0], upper=ac.aero['theta_flaps'][-1], dynamic=True, include_timeseries=True, opt=True, ref=10.)                
                # self.phases['liftoff'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=15, dynamic=True, include_timeseries=True, opt=True)
            # else:
            self.phases['liftoff'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, dynamic=True, include_timeseries=True)
            self.phases['liftoff'].add_parameter('theta_slats', targets='theta_slats', units='deg', val=settings.theta_slats, dynamic=True, include_timeseries=True)

        # Phase 4: vnrs phase
        if 'vnrs' in self.phase_name_lst:
            opts = {'phase': 'vnrs', 'ac': ac, 'engine': engine, 'settings': settings, 'objective': objective}
            self.phases['vnrs'] = dm.Phase(ode_class=TrajectoryODE, transcription=self.transcription_phases[3], ode_init_kwargs=opts)
            self.phases['vnrs'].set_time_options(initial_bounds=(20, 150), duration_bounds=(0, 500), initial_ref=100., duration_ref=100.)            
            if trajectory_mode == 'flyover':
                self.phases['vnrs'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=True, ref=10000.)
                self.phases['vnrs'].add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=True, fix_final=False, ref=1000.)
            elif trajectory_mode == 'cutback':
                self.phases['vnrs'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=10000.)
                self.phases['vnrs'].add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=True, fix_final=True, ref=1000.)
            self.phases['vnrs'].add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
            self.phases['vnrs'].add_state('gamma', rate_source='flight_dynamics.gamma_dot', units='deg', fix_initial=False, fix_final=False, ref=10.)
            self.phases['vnrs'].add_control('alpha', targets='alpha', units='deg', lower=5., upper=ac.aero['alpha'][-1], rate_continuity=True, rate_continuity_scaler=1.0, rate2_continuity=False, opt=True, ref=10.)
            self.phases['vnrs'].add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_size[3]-1, order=3, solve_segments=False, compressed=True), subset='state_input')
            self.phases['vnrs'].add_path_constraint(name='flight_dynamics.v_dot', lower=0., units='m/s**2')
            self.phases['vnrs'].add_path_constraint(name='gamma', lower=0., units='deg', ref=10.)
            # PTCB
            if objective == 'noise' and settings.PTCB:
                self.phases['vnrs'].add_control('TS', targets='propulsion.TS', units=None, upper=settings.TS_to, lower=TS_min, val=TS_min, opt=True, rate_continuity=True, rate2_continuity=False, ref=1.)
                self.phases['vnrs'].add_path_constraint(name='TS', lower=TS_min, upper=1, units=None, ref=1.)
            else:
                self.phases['vnrs'].add_parameter('TS', targets='propulsion.TS', units=None, val=settings.TS_vnrs, dynamic=True, include_timeseries=True)
            # PHLD
            if objective == 'noise' and settings.PHLD:
                self.phases['vnrs'].add_control('theta_flaps', targets='theta_flaps', units='deg', val=0., lower=ac.aero['theta_flaps'][0], upper=ac.aero['theta_flaps'][-1], opt=True, rate_continuity=True, rate2_continuity=False, ref=1.)
                # self.phases['vnrs'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=0., lower=ac.aero['theta_flaps'][0], upper=ac.aero['theta_flaps'][-1], dynamic=True, include_timeseries=True, opt=True)
            else:
                self.phases['vnrs'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, dynamic=True, include_timeseries=True)
            self.phases['vnrs'].add_parameter('theta_slats', targets='theta_slats', units='deg', val=settings.theta_slats, dynamic=True, include_timeseries=True)

        # Phase 5: cutback phase
        if 'cutback' in self.phase_name_lst:
            opts = {'phase': 'cutback', 'ac': ac, 'engine': engine, 'settings': settings, 'objective': objective}
            self.phases['cutback'] = dm.Phase(ode_class=TrajectoryODE, transcription=self.transcription_phases[4], ode_init_kwargs=opts)
            self.phases['cutback'].set_time_options(initial_bounds=(20, 150), duration_bounds=(0, 500), initial_ref=100., duration_ref=100.)
            if trajectory_mode == 'flyover':
                self.phases['cutback'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=True, fix_final=False, ref=10000.)
                self.phases['cutback'].add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=False, fix_final=True, ref=1000.)
            elif trajectory_mode == 'cutback':
                self.phases['cutback'].add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=10000.)
                self.phases['cutback'].add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=True, fix_final=True, ref=1000.)
            self.phases['cutback'].add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
            self.phases['cutback'].add_state('gamma', rate_source='flight_dynamics.gamma_dot', units='deg', fix_initial=False, fix_final=False, ref=10.)
            self.phases['cutback'].add_control('alpha', targets='alpha', units='deg', lower=ac.aero['alpha'][0], upper=ac.aero['alpha'][-1], rate_continuity=True, rate_continuity_scaler=1.0, rate2_continuity=False, opt=True, ref=10.)
            self.phases['cutback'].add_path_constraint(name='flight_dynamics.v_dot', lower=0., units='m/s**2')
            # self.phases['cutback'].add_path_constraint(name='flight_dynamics.gamma_dot', upper=0., units='deg/s')
            self.phases['cutback'].add_boundary_constraint('v', loc='final', upper=ac.v_max, ref=100., units='m/s')
            self.phases['cutback'].add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_size[4]-1, order=3, solve_segments=False, compressed=True), subset='state_input')
            # PTCB
            self.phases['cutback'].add_parameter('TS', targets='propulsion.TS', units=None, val=TS_min, dynamic=True, include_timeseries=True)
            # PHLD
            if objective == 'noise' and settings.PHLD:
                self.phases['cutback'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=0., lower=ac.aero['theta_flaps'][0], upper=ac.aero['theta_flaps'][-1], dynamic=True, include_timeseries=True, opt=True, ref=10.)
            else:
                self.phases['cutback'].add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=settings.theta_flaps, dynamic=True, include_timeseries=True)
            self.phases['cutback'].add_parameter('theta_slats', targets='theta_slats', units='deg', val=settings.theta_slats, dynamic=True, include_timeseries=True)

        # Add outputs to timeseries for each phase
        for j, phase_name in enumerate(self.phase_name_lst):
            for i in np.arange(len(engine_var)):
                self.phases[phase_name].add_timeseries_output('propulsion.'+ engine_var[i], timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('aerodynamics.M_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('p_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('rho_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('I_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('drho_0_dz', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('T_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('c_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('mu_0', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('propulsion.W_f', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('emissions.mdot_NOx', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('emissions.EINOx', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('flight_dynamics.y', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('flight_dynamics.n', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('flight_dynamics.I_landing_gear', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('flight_dynamics.eas', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('flight_dynamics.n', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('aerodynamics.L', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('aerodynamics.D', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('propulsion.F_n', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('propulsion.W_f', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('clcd.c_l', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('clcd.c_l_max', timeseries='interpolated')
            self.phases[phase_name].add_timeseries_output('clcd.c_d', timeseries='interpolated')

        # Add phases to the trajectory
        for phase_name in self.phase_name_lst:
            traj.add_phase(phase_name, self.phases[phase_name])

        # Link phases
        if 'rotation' in self.phase_name_lst:
            traj.link_phases(phases=['groundroll', 'rotation'], vars=['time', 'x', 'v', 'alpha'])
            # if objective == 'noise' and settings.PHLD:
                # traj.add_linkage_constraint(phase_a='groundroll', phase_b='rotation', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'liftoff' in self.phase_name_lst:
            traj.link_phases(phases=['rotation', 'liftoff'], vars=['time', 'x', 'z', 'v', 'alpha', 'gamma'])
            # if objective == 'noise' and settings.PHLD:
                # traj.add_linkage_constraint(phase_a='rotation', phase_b='liftoff', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')
        
        if 'vnrs' in self.phase_name_lst:
            traj.link_phases(phases=['liftoff', 'vnrs'],  vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings.PTCB:
                traj.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='TS', var_b='TS', loc_a='final', loc_b='initial')
            if objective == 'noise' and settings.PHLD:
                traj.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'cutback' in self.phase_name_lst:
            if trajectory_mode == 'flyover':
                traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'z', 'v', 'alpha', 'gamma'])
            elif trajectory_mode == 'cutback':
                traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'x', 'v', 'alpha', 'gamma'])
            # if objective == 'noise' and settings.PTCB:
                # traj.add_linkage_constraint(phase_a='vnrs', phase_b='cutback', var_a='TS', var_b='TS', loc_a='final', loc_b='initial')
            if objective == 'noise' and settings.PHLD:
                traj.add_linkage_constraint(phase_a='vnrs', phase_b='cutback', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        # Mux trajectory variables
        mux_t = problem.model.add_subsystem(name='trajectory', subsys=Mux(size_inputs=np.array(self.phase_size), size_output=self.trajectory_size))
        var   = ['time', 'x', 'y', 'z', 'v', 'M_0', 'alpha', 'gamma', 'TS', 'I_landing_gear', 'theta_flaps', 'theta_slats','F_n', 'L', 'D', 'eas', 'n', 'p_0','rho_0','drho_0_dz','T_0','c_0','mu_0', 'I_0', 'W_f', 'mdot_NOx', 'EINOx', 'c_l', 'c_d', 'c_l_max']
        units = ['s', 'm', 'm', 'm', 'm/s', None, 'deg', 'deg', None, None, 'deg', 'deg', 'N', 'N', 'N', 'm/s', None, 'Pa', 'kg/m**3', 'kg/m**4', 'K', 'm/s', 'kg/m/s', 'kg/m**2/s', 'kg/s', 'kg/s', None, None, None, None]
        for i in np.arange(len(var)):
            # Add the variables to the trajectory mux
            if var[i] == 'time':
                mux_t.add_var('t_s', units=units[i])
            else:
                mux_t.add_var(var[i], units=units[i])

            # Connect phase variables to trajectory mux
            if var[i] == 'time':
                for j, phase_name in enumerate(self.phase_name_lst):
                    problem.model.connect('phases.' + phase_name + '.interpolated.' + var[i], 'trajectory.t_s_' + str(j))

            elif var[i] in ['x', 'v']:
                for j, phase_name in enumerate(self.phase_name_lst):
                    problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['z', 'gamma']:
                for j, phase_name in enumerate(self.phase_name_lst):
                    if phase_name in {'groundroll','rotation'}:
                        problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                    else:
                        problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['alpha']:
                for j, phase_name in enumerate(self.phase_name_lst):
                    if phase_name in {'groundroll', 'rotation'}:
                        problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var[i],'trajectory.' + var[i] + '_' + str(j))
                    else:
                        problem.model.connect('phases.' + phase_name + '.interpolated.controls:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['TS']:
                if objective == 'noise' and settings.PTCB:
                    for j, phase_name in enumerate(self.phase_name_lst):
                        if phase_name in ['groundroll', 'rotation', 'liftoff']:
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                        elif phase_name == 'vnrs':
                            problem.model.connect('phases.' + phase_name + '.interpolated.controls:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                        elif phase_name == 'cutback':
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                else:
                    for j, phase_name in enumerate(self.phase_name_lst):
                        if phase_name in ['groundroll', 'rotation', 'liftoff', 'vnrs']:
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                        elif phase_name == 'cutback':
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['theta_slats']:
                for j, phase_name in enumerate(self.phase_name_lst):
                    problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['theta_flaps']:
                if objective == 'noise' and settings.PHLD:
                    for j, phase_name in enumerate(self.phase_name_lst):
                        if phase_name == 'vnrs':
                            problem.model.connect('phases.' + phase_name + '.interpolated.controls:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                        else:
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))
                else:
                    for j, phase_name in enumerate(self.phase_name_lst):
                        problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var[i], 'trajectory.' + var[i] + '_' + str(j))

            elif var[i] in ['y', 'F_n', 'L', 'D', 'eas', 'n', 'I_landing_gear','M_0', 'p_0','rho_0','drho_0_dz','T_0','c_0','c_bar','mu_0', 'I_0', 'W_f', 'mdot_NOx', 'EINOx', 'c_l', 'c_d', 'c_l_max']:
                for j, phase_name in enumerate(self.phase_name_lst):
                    problem.model.connect('phases.' + phase_name + '.interpolated.' + var[i], 'trajectory.' + var[i] + '_' + str(j))

        # Mux engine variables
        mux_e = problem.model.add_subsystem(name='engine', subsys=Mux(size_inputs=np.array(self.phase_size), size_output=self.trajectory_size))
        for i in np.arange(len(engine_var)):
            mux_e.add_var(engine_var[i], units=engine_var_units[i])
            for j, phase_name in enumerate(self.phase_name_lst):
                problem.model.connect('phases.' + phase_name + '.interpolated.' + engine_var[i], 'engine.' + engine_var[i] + '_' + str(j))

        return None

    def compute(self, problem: om.Problem, settings: Settings, ac: Aircraft, run_driver: bool, init_trajectory: om.Problem, trajectory_mode: str, objective: str) -> None:
        """
        Run trajectory initial guess with minimal time to climb.

        :param problem: openmdao problem
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
        :param run_driver: flag to enable run_driver setting for dymos run_model function
        :type run_driver: bool
        :param init_trajectory: initialization trajectory
        :type init_trajectory: om.Problem
        :param objective: optimization objective
        :type objective: str

        :return: None
        """

        # Add objective for trajectory model
        if objective == None:
            # No optimization objective required; problem is run with run_driver = False
            pass

        elif objective == 'x_end':
            problem.model.add_objective('trajectory.x', index=-1, ref=1000.)
        
        elif objective == 't_end':
            problem.model.add_objective('trajectory.t_s', index=-1, ref=1000.)
        
        elif objective == 'noise':
            # Optimization bjective is set in the noise.setup_trajectory_noise() method
            pass

        elif objective == 'x_takeoff_x_end':
            problem.model.add_subsystem('combined', om.ExecComp('objective = x_to/2000 + x_end/7000', 
                                                                x_to={'val': 0., 'units': 'm'},
                                                                x_end={'val': 0., 'units': 'm'}), 
                                                                promotes=['objective'])
            problem.model.connect('phases.liftoff.timeseries.states:x' , 'combined.x_to', src_indices=[-1])
            problem.model.connect('trajectory.x', 'combined.x_end', src_indices=[-1])
            problem.model.add_objective('objective', scaler=10.)

        else: 
            raise ValueError('Invalid control objective specified.')

        # Run the openMDAO problem setup
        problem.setup(force_alloc_complex=True)

        # Attach a recorder to the problem to save model data
        if settings.save_results:
            problem.add_recorder(om.SqliteRecorder(settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_directory_name + '/' + settings.output_file_name))

        # Set initial guess for the trajectory problem
        if init_trajectory is None:

            # Phase 1: groundroll
            if 'groundroll' in self.phase_name_lst:
                problem['phases.groundroll.t_initial'] = 0.0
                problem['phases.groundroll.t_duration'] = 30.0
                problem['phases.groundroll.states:x'] = self.phases['groundroll'].interp(ys=[0, 1000], nodes='state_input')
                problem['phases.groundroll.states:v'] = self.phases['groundroll'].interp(ys=[0.0, 60], nodes='state_input')
                problem['phases.groundroll.states:alpha'] = self.phases['groundroll'].interp(ys=[ac.alpha_0, ac.alpha_0], nodes='state_input')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                problem['phases.rotation.t_initial'] = 30.0
                problem['phases.rotation.t_duration'] = 10.0
                problem['phases.rotation.states:x'] = self.phases['rotation'].interp(ys=[1500, 2000], nodes='state_input')
                problem['phases.rotation.states:v'] = self.phases['rotation'].interp(ys=[100, 110.], nodes='state_input')
                problem['phases.rotation.states:alpha'] = self.phases['rotation'].interp(ys=[ac.alpha_0, 15*np.pi/180.], nodes='state_input')

            # Phase 3: lift-off
            if 'liftoff' in self.phase_name_lst:
                if trajectory_mode == 'flyover':
                    z_cutback_guess = 500.
                elif trajectory_mode == 'cutback':
                    z_cutback_guess = settings.z_cutback
                problem['phases.liftoff.t_initial'] = 40.0
                problem['phases.liftoff.t_duration'] = 2.
                problem['phases.liftoff.states:x'] = self.phases['liftoff'].interp(ys=[2000., 3500.], nodes='state_input')
                problem['phases.liftoff.states:z'] = self.phases['liftoff'].interp(ys=[0., 35*0.3048], nodes='state_input')
                problem['phases.liftoff.states:v'] = self.phases['liftoff'].interp(ys=[110., 110.], nodes='state_input')
                problem['phases.liftoff.states:gamma'] = self.phases['liftoff'].interp(ys=[0, 4.], nodes='state_input')
                problem['phases.liftoff.controls:alpha'] = self.phases['liftoff'].interp(ys=[15., 15.], nodes='control_input')

            # # Phase 4: vnrs 
            if 'vnrs' in self.phase_name_lst:
                problem['phases.vnrs.t_initial'] = 50.0
                problem['phases.vnrs.t_duration'] = 50.0
                problem['phases.vnrs.states:x'] = self.phases['vnrs'].interp(ys=[3500., 6501.], nodes='state_input')
                problem['phases.vnrs.states:z'] = self.phases['vnrs'].interp(ys=[35*0.3048, z_cutback_guess], nodes='state_input')
                problem['phases.vnrs.states:v'] = self.phases['vnrs'].interp(ys=[110., 110.], nodes='state_input')
                problem['phases.vnrs.states:gamma'] = self.phases['vnrs'].interp(ys=[4., 15.], nodes='state_input')
                problem['phases.vnrs.controls:alpha'] = self.phases['vnrs'].interp(ys=[15., 15.], nodes='control_input')
                
            # Phase 5: cutback
            if 'cutback' in self.phase_name_lst:
                problem['phases.cutback.t_initial'] = 100.0
                problem['phases.cutback.t_duration'] = 50.0
                problem['phases.cutback.states:x'] = self.phases['cutback'].interp(ys=[6501., 20000.], nodes='state_input')
                problem['phases.cutback.states:z'] = self.phases['cutback'].interp(ys=[z_cutback_guess, ac.z_max], nodes='state_input')
                problem['phases.cutback.states:v'] = self.phases['cutback'].interp(ys=[110., 110.], nodes='state_input')
                problem['phases.cutback.states:gamma'] = self.phases['cutback'].interp(ys=[15, 15.], nodes='state_input')
                problem['phases.cutback.controls:alpha'] = self.phases['cutback'].interp(ys=[15., 15.], nodes='control_input')
                
        else:
            # Phase 1: groundroll 
            if 'groundroll' in self.phase_name_lst:
                problem['phases.groundroll.t_initial'] = init_trajectory.get_val('phases.groundroll.t_initial')
                problem['phases.groundroll.t_duration'] = init_trajectory.get_val('phases.groundroll.t_duration')
                problem['phases.groundroll.timeseries.time'] = init_trajectory.get_val('phases.groundroll.timeseries.time')
                problem['phases.groundroll.states:x'] = init_trajectory.get_val('phases.groundroll.states:x')
                problem['phases.groundroll.states:v'] = init_trajectory.get_val('phases.groundroll.states:v')
                problem['phases.groundroll.states:alpha'] = init_trajectory.get_val('phases.groundroll.states:alpha')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                problem['phases.rotation.t_initial'] = init_trajectory.get_val('phases.rotation.t_initial')
                problem['phases.rotation.t_duration'] = init_trajectory.get_val('phases.rotation.t_duration')
                problem['phases.rotation.timeseries.time'] = init_trajectory.get_val('phases.rotation.timeseries.time')
                problem['phases.rotation.states:x'] = init_trajectory.get_val('phases.rotation.states:x')
                problem['phases.rotation.states:v'] = init_trajectory.get_val('phases.rotation.states:v')
                problem['phases.rotation.states:alpha'] = init_trajectory.get_val('phases.rotation.states:alpha')

            # Phase 3-5: liftoff-cutback
            for j, phase_name in enumerate(self.phase_name_lst[3:]):
                problem['phases.' + phase_name + '.t_initial'] = init_trajectory.get_val('phases.' + phase_name + '.t_initial')
                problem['phases.' + phase_name + '.t_duration'] = init_trajectory.get_val('phases.' + phase_name + '.t_duration')
                problem['phases.' + phase_name + '.timeseries.time'] = init_trajectory.get_val('phases.' + phase_name + '.timeseries.time')
                problem['phases.' + phase_name + '.states:x'] = init_trajectory.get_val('phases.' + phase_name + '.states:x')
                problem['phases.' + phase_name + '.states:z'] = init_trajectory.get_val('phases.' + phase_name + '.states:z')
                problem['phases.' + phase_name + '.states:v'] = init_trajectory.get_val('phases.' + phase_name + '.states:v')
                problem['phases.' + phase_name + '.states:gamma'] = init_trajectory.get_val('phases.' + phase_name + '.states:gamma')
                problem['phases.' + phase_name + '.controls:alpha'] = init_trajectory.get_val('phases.' + phase_name + '.controls:alpha')
                # if phase_name == 'vnrs' and settings.PTCB:
                    # problem['phases.' + phase_name + '.controls:TS'] = init_trajectory.get_val('phases.' + phase_name + '.controls:TS')

        # Run problem
        dm.run_problem(problem, run_driver=run_driver)

        # Save the results
        if settings.save_results:
            problem.record(case_name=settings.ac_name)

        # Write output
        return None

    @staticmethod
    def check_convergence(settings: Settings, filename: str) -> bool:
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
        file_ipopt = open(settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_directory_name + '/' + filename, 'r')
        ipopt = file_ipopt.readlines()
        file_ipopt.close()

        # Check if convergence summary excel file exists
        cnvg_file_name = settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_directory_name + '/' + 'Convergence.csv'
        if not os.path.isfile(cnvg_file_name):
            file_cvg = open(cnvg_file_name, 'w')
            file_cvg.writelines("Trajectory name , Execution date/time,  Converged")
        else:
            file_cvg = open(cnvg_file_name, 'a')

        # Write convergence output to file
        # file = open(cnvg_file_name, 'a')
        if ipopt[-1] in {'EXIT: Optimal Solution Found.\n', 'EXIT: Solved To Acceptable Level.\n'}:
            file_cvg.writelines("\n" + settings.output_file_name + ", " + str(dt.datetime.now()) + ", Converged")
            converged = True
        else:
            file_cvg.writelines("\n" + settings.output_file_name + ", " + str(dt.datetime.now()) + ", Not converged")
            converged = False
        file_cvg.close()

        return converged


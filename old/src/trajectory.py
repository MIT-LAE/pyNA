import pdb
import os
import datetime as dt
import openmdao.api as om
import dymos as dm
import numpy as np
import matplotlib.pyplot as plt
from pyNA.src.airframe import Airframe
from pyNA.src.trajectory_src.groundroll import GroundRoll
from pyNA.src.trajectory_src.rotation import Rotation
from pyNA.src.trajectory_src.liftoff import LiftOff
from pyNA.src.trajectory_src.vnrs import Vnrs
from pyNA.src.trajectory_src.cutback import CutBack
from pyNA.src.trajectory_src.take_off_phase_ode import TakeOffPhaseODE
from pyNA.src.trajectory_src.mux import Mux

if os.environ['pyna_language']=='julia':
    import julia.Main as julia
    from julia.OpenMDAO import make_component
    src_path = os.path.dirname(os.path.abspath(__file__))
    julia.include(src_path + "/noise_src_jl/noise_model.jl")
elif os.environ['pyna_language']=='python':
    from pyNA.src.noise_src_py.noise_model import NoiseModel


class Trajectory(om.Problem):

    def __init__(self, pyna_directory, case_name, language, output_directory_name, output_file_name, model=None, driver=None, comm=None, name=None, **options):
        super().__init__(model, driver, comm, name, **options)

        self.pyna_directory = pyna_directory
        self.case_name = case_name
        self.language = language
        self.output_directory_name = output_directory_name
        self.output_file_name = output_file_name

        self.phase_name_lst = ['groundroll', 'rotation', 'liftoff', 'vnrs', 'cutback']

        self.transcription = dict()
        for phase_name in self.phase_name_lst:
            self.transcription[phase_name] = dict()

            if phase_name == 'groundroll':
                self.transcription[phase_name]['num_segments'] = 3
                self.transcription[phase_name]['order'] = 3
            elif phase_name == 'rotation':
                self.transcription[phase_name]['num_segments'] = 3
                self.transcription[phase_name]['order'] = 3
            elif phase_name == 'liftoff':
                self.transcription[phase_name]['num_segments'] = 4
                self.transcription[phase_name]['order'] = 3
            elif phase_name == 'vnrs':
                self.transcription[phase_name]['num_segments'] = 8
                self.transcription[phase_name]['order'] = 3
            elif phase_name == 'cutback':
                self.transcription[phase_name]['num_segments'] = 10
                self.transcription[phase_name]['order'] = 3

            self.transcription[phase_name]['grid'] = dm.GaussLobatto(num_segments=self.transcription[phase_name]['num_segments'], order=self.transcription[phase_name]['order'], compressed=True, solve_segments=False)
            self.transcription[phase_name]['grid'].init_grid()

    def get_mux_input_output_size(self) -> None:
        """
        Compute vector size of the muxed trajectory.

        """

        # List input sizes
        self.mux_input_size_array = []
        for phase_name in self.phase_name_lst:

            if phase_name == 'groundroll':
                self.mux_input_size_array.append(self.groundroll.phase_target_size)
            elif phase_name == 'rotation':
                self.mux_input_size_array.append(self.rotation.phase_target_size)
            elif phase_name == 'liftoff':
                self.mux_input_size_array.append(self.liftoff.phase_target_size)
            elif phase_name == 'vnrs':
                self.mux_input_size_array.append(self.vnrs.phase_target_size)
            elif phase_name == 'cutback':
                self.mux_input_size_array.append(self.cutback.phase_target_size)

        # List output sizes
        self.mux_output_size = 0
        for i, phase_name in enumerate(self.phase_name_lst):
            if phase_name == 'groundroll':
                if i+1 == len(self.phase_name_lst):
                    self.mux_output_size += self.groundroll.phase_target_size
                else:
                    self.mux_output_size += self.groundroll.phase_target_size - 1
            elif phase_name == 'rotation':
                if i+1 == len(self.phase_name_lst):
                    self.mux_output_size += self.rotation.phase_target_size
                else:
                    self.mux_output_size += self.rotation.phase_target_size - 1
            elif phase_name == 'liftoff':
                if i+1 == len(self.phase_name_lst):
                    self.mux_output_size += self.liftoff.phase_target_size
                else:
                    self.mux_output_size += self.liftoff.phase_target_size - 1
            elif phase_name == 'vnrs':
                if i+1 == len(self.phase_name_lst):
                    self.mux_output_size += self.vnrs.phase_target_size
                else:
                    self.mux_output_size += self.vnrs.phase_target_size - 1
            elif phase_name == 'cutback':
                if i+1 == len(self.phase_name_lst):
                    self.mux_output_size += self.cutback.phase_target_size
                else:
                    self.mux_output_size += self.cutback.phase_target_size - 1
        
        return None

    def set_ipopt_settings(self, objective, tolerance, max_iter) -> None:
        
        # Set solver settings for the problem
        self.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        self.driver.opt_settings['print_level'] = 5
        self.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

        self.driver.declare_coloring(tol=1e-12)
        self.model.linear_solver = om.LinearRunOnce()
        self.driver.opt_settings['output_file'] = self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/IPOPT_trajectory_convergence.out'

        if objective == 'noise':
            self.driver.opt_settings['tol'] = 1e-3
            self.driver.opt_settings['acceptable_tol'] = 1e-1
        else:
            self.driver.opt_settings['tol'] = tolerance
            self.driver.opt_settings['acceptable_tol'] = 1e-2

        self.driver.opt_settings['max_iter'] = max_iter
        self.driver.opt_settings['mu_strategy'] = 'adaptive'
        self.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        self.driver.opt_settings['mu_init'] = 0.01
        self.driver.opt_settings['constr_viol_tol'] = 1e-3
        self.driver.opt_settings['compl_inf_tol'] = 1e-3
        self.driver.opt_settings['acceptable_iter'] = 0
        self.driver.opt_settings['acceptable_constr_viol_tol'] = 1e-1
        self.driver.opt_settings['acceptable_compl_inf_tol'] = 1e-1
        self.driver.opt_settings['acceptable_obj_change_tol'] = 1e-1

        return None

    @staticmethod
    def get_trajectory_var_lst(atmosphere_type) -> dict():
        
        trajectory_var = dict()
        if atmosphere_type == 'stratified':
            trajectory_var['x'] = 'm'
            trajectory_var['y'] = 'm'
            trajectory_var['z'] = 'm'
            trajectory_var['v'] = 'm/s'
            trajectory_var['alpha'] = 'deg'
            trajectory_var['gamma'] = 'deg'
            trajectory_var['time'] = 's'
            trajectory_var['M_0'] = None
            trajectory_var['TS'] = None
            trajectory_var['c_0'] = 'm/s'
            trajectory_var['T_0'] = 'K'
            trajectory_var['rho_0'] = 'kg/m**3'
            trajectory_var['p_0'] = 'Pa'
            trajectory_var['mu_0'] = 'kg/m/s'
            trajectory_var['I_0'] = 'kg/m**2/s'
            trajectory_var['I_landing_gear'] = None
            trajectory_var['theta_flaps'] = 'deg'
            trajectory_var['theta_slats'] = 'deg'
            trajectory_var['L'] = 'N'
            trajectory_var['D'] = 'N'
            trajectory_var['c_l'] = None
            trajectory_var['c_d'] = None
            trajectory_var['c_l_max'] = None
            trajectory_var['n'] = None
            trajectory_var['mdot_NOx'] = 'kg/s'
            trajectory_var['EINOx'] = None
        else:
            trajectory_var['x'] = 'm'
            trajectory_var['y'] = 'm'
            trajectory_var['z'] = 'm'
            trajectory_var['v'] = 'm/s'
            trajectory_var['alpha'] = 'deg'
            trajectory_var['gamma'] = 'deg'
            trajectory_var['time'] = 's'
            trajectory_var['M_0'] = None
            trajectory_var['TS'] = None
            trajectory_var['I_landing_gear'] = None
            trajectory_var['theta_flaps'] = 'deg'
            trajectory_var['theta_slats'] = 'deg'
            trajectory_var['L'] = 'N'
            trajectory_var['D'] = 'N'
            trajectory_var['c_l'] = None
            trajectory_var['c_d'] = None
            trajectory_var['c_l_max'] = None
            trajectory_var['n'] = None
            trajectory_var['mdot_NOx'] = 'kg/s'
            trajectory_var['EINOx'] = None

        return trajectory_var

    def create_trajectory(self, airframe, engine, sealevel_atmosphere, k_rot, v_max, TS_to, TS_vnrs, TS_cb, TS_min=0.4, theta_flaps=10., theta_flaps_cb=10., theta_slats=-6, atmosphere_type='stratified', atmosphere_dT=10.0169, pkrot=False, ptcb=False, phld=False, objective='t_end', trajectory_mode='cutback') -> None:

        # Add dymos trajectory to the problem
        self.traj = dm.Trajectory()
        self.model.add_subsystem('phases', self.traj)

        # Create the trajectory phases 
        for phase_name in self.phase_name_lst:
            opts = {'phase': phase_name, 'airframe': airframe, 'engine': engine, 'sealevel_atmosphere': sealevel_atmosphere, 'atmosphere_dT': atmosphere_dT, 'atmosphere_type': atmosphere_type, 'objective': objective, 'case_name': self.case_name, 'output_directory_name': self.output_directory_name}
            
            if phase_name == 'groundroll':
                self.groundroll = GroundRoll(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.groundroll.create(airframe, engine, pkrot, phld, TS_to, k_rot, theta_flaps, theta_slats, objective, atmosphere_type)
                self.traj.add_phase(phase_name, self.groundroll)
            
            elif phase_name == 'rotation':
                self.rotation = Rotation(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.rotation.create(airframe, engine, phld, TS_to, theta_flaps, theta_slats, objective, atmosphere_type)
                self.traj.add_phase(phase_name, self.rotation)

            elif phase_name == 'liftoff':
                self.liftoff = LiftOff(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.liftoff.create(airframe, engine, phld, TS_to, theta_flaps, theta_slats, objective, atmosphere_type)
                self.traj.add_phase(phase_name, self.liftoff)

            elif phase_name == 'vnrs':
                self.vnrs = Vnrs(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.vnrs.create(airframe, engine, ptcb, phld, TS_vnrs, TS_min, theta_flaps_cb, theta_slats, trajectory_mode, objective, atmosphere_type)
                self.traj.add_phase(phase_name, self.vnrs)
                
            elif phase_name == 'cutback':
                self.cutback = CutBack(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.cutback.create(airframe, engine, phld, v_max, TS_cb, theta_flaps_cb, theta_slats, trajectory_mode, objective, atmosphere_type)
                self.traj.add_phase(phase_name, self.cutback)        

        # Link phases
        if 'rotation' in self.phase_name_lst:
            self.traj.link_phases(phases=['groundroll', 'rotation'], vars=['time', 'x', 'v', 'alpha'])    
            if objective == 'noise' and phld:
                self.traj.add_linkage_constraint(phase_a='groundroll', phase_b='rotation', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'liftoff' in self.phase_name_lst:
            self.traj.link_phases(phases=['rotation', 'liftoff'], vars=['time', 'x', 'z', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and phld:
                self.traj.add_linkage_constraint(phase_a='rotation', phase_b='liftoff', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'vnrs' in self.phase_name_lst:
            self.traj.link_phases(phases=['liftoff', 'vnrs'],  vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and ptcb:
                self.traj.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='TS', var_b='TS', loc_a='final', loc_b='initial')

        if 'cutback' in self.phase_name_lst:
            if trajectory_mode == 'flyover':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'z', 'v', 'alpha', 'gamma'])
            elif trajectory_mode == 'cutback':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and ptcb:
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

    def create_noise(self, settings, data, sealevel_atmosphere, airframe, n_t:int, objective:str, mode:str) -> None:

        """
        Setup model for computing noise along computed trajectory.

        :param airframe: aircraft parameters
        :type airframe: Airframe
        :param n_t: number of time steps in trajectory
        :type n_t: np.int
        :param objective: optimization objective
        :type objective: str

        :return: None
        """

        if self.language == 'python':
            self.model.add_subsystem(name='noise',
                                        subsys=NoiseModel(settings=settings, data=data, airframe=airframe, n_t=n_t, mode=mode), 
                                        promotes_inputs=[],
                                        promotes_outputs=[])
        
            # Create connections from trajectory group
            self.model.connect('trajectory.c_0', 'noise.normalize_engine.c_0')
            self.model.connect('trajectory.T_0', 'noise.normalize_engine.T_0')
            self.model.connect('trajectory.p_0', 'noise.normalize_engine.p_0')
            self.model.connect('trajectory.rho_0', 'noise.normalize_engine.rho_0')
            if settings['jet_mixing_source'] and settings['jet_shock_source'] == False:
                self.model.connect('engine.V_j', 'noise.normalize_engine.V_j')
                self.model.connect('engine.rho_j', 'noise.normalize_engine.rho_j')
                self.model.connect('engine.A_j', 'noise.normalize_engine.A_j')
                self.model.connect('engine.Tt_j', 'noise.normalize_engine.Tt_j')
            elif settings['jet_shock_source'] and settings['jet_mixing_source'] == False:
                self.model.connect('engine.V_j', 'noise.normalize_engine.V_j')
                self.model.connect('engine.A_j', 'noise.normalize_engine.A_j')
                self.model.connect('engine.Tt_j', 'noise.normalize_engine.Tt_j')
                self.model.connect('engine.M_j', 'noise.source.M_j')
                self.model.connect('engine.A_j', 'noise.A_j')
                self.model.connect('engine.Tt_j', 'noise.Tt_j')
                self.model.connect('engine.M_j', 'noise.M_j')
            elif settings['jet_shock_source'] and settings['jet_mixing_source']:
                self.model.connect('engine.V_j', 'noise.normalize_engine.V_j')
                self.model.connect('engine.rho_j', 'noise.normalize_engine.rho_j')
                self.model.connect('engine.A_j', 'noise.normalize_engine.A_j')
                self.model.connect('engine.Tt_j', 'noise.normalize_engine.Tt_j')
                self.model.connect('engine.M_j', 'noise.source.M_j')
            if settings['core_source']:
                if settings['core_turbine_attenuation_method'] == 'ge':
                    self.model.connect('engine.mdoti_c', 'noise.normalize_engine.mdoti_c')
                    self.model.connect('engine.Tti_c', 'noise.normalize_engine.Tti_c')
                    self.model.connect('engine.Ttj_c', 'noise.normalize_engine.Ttj_c')
                    self.model.connect('engine.Pti_c', 'noise.normalize_engine.Pti_c')
                    self.model.connect('engine.DTt_des_c', 'noise.normalize_engine.DTt_des_c')
                elif settings['core_turbine_attenuation_method'] == 'pw':
                    self.model.connect('engine.mdoti_c', 'noise.normalize_engine.mdoti_c')
                    self.model.connect('engine.Tti_c', 'noise.normalize_engine.Tti_c')
                    self.model.connect('engine.Ttj_c', 'noise.normalize_engine.Ttj_c')
                    self.model.connect('engine.Pti_c', 'noise.normalize_engine.Pti_c')
                    self.model.connect('engine.rho_te_c', 'noise.normalize_engine.rho_te_c')
                    self.model.connect('engine.c_te_c', 'noise.normalize_engine.c_te_c')
                    self.model.connect('engine.rho_ti_c', 'noise.normalize_engine.rho_ti_c')
                    self.model.connect('engine.c_ti_c', 'noise.normalize_engine.c_ti_c')
            if settings['fan_inlet_source'] or settings['fan_discharge_source']:
                self.model.connect('engine.mdot_f', 'noise.normalize_engine.mdot_f')
                self.model.connect('engine.N_f', 'noise.normalize_engine.N_f')
                self.model.connect('engine.DTt_f', 'noise.normalize_engine.DTt_f')
                self.model.connect('engine.A_f', 'noise.normalize_engine.A_f')
                self.model.connect('engine.d_f', 'noise.normalize_engine.d_f')

            self.model.connect('trajectory.x', 'noise.geometry.x')
            self.model.connect('trajectory.y', 'noise.geometry.y')
            self.model.connect('trajectory.z', 'noise.geometry.z')
            self.model.connect('trajectory.alpha', 'noise.geometry.alpha')
            self.model.connect('trajectory.gamma', 'noise.geometry.gamma')
            self.model.connect('trajectory.c_0', 'noise.geometry.c_0')
            self.model.connect('trajectory.T_0', 'noise.geometry.T_0')
            self.model.connect('trajectory.t_s', 'noise.geometry.t_s')
            
            self.model.connect('trajectory.TS', 'noise.source.TS')
            self.model.connect('trajectory.M_0', 'noise.source.M_0')
            self.model.connect('trajectory.c_0', 'noise.source.c_0')
            self.model.connect('trajectory.rho_0', 'noise.source.rho_0')
            self.model.connect('trajectory.mu_0', 'noise.source.mu_0')
            self.model.connect('trajectory.T_0', 'noise.source.T_0')
            
            if settings['airframe_source']:
                self.model.connect('trajectory.I_landing_gear', 'noise.source.I_landing_gear')
                self.model.connect('trajectory.theta_flaps', 'noise.source.theta_flaps')

            self.model.connect('trajectory.x', 'noise.propagation.x')
            self.model.connect('trajectory.z', 'noise.propagation.z')
            self.model.connect('trajectory.rho_0', 'noise.propagation.rho_0')
            self.model.connect('trajectory.I_0', 'noise.propagation.I_0')
            
            self.model.connect('trajectory.c_0', 'noise.levels.c_0')
            self.model.connect('trajectory.rho_0', 'noise.levels.rho_0')

        elif self.language == 'julia':
            self.model.add_subsystem(name='noise',
                                        subsys=make_component(julia.NoiseModel(settings, data, sealevel_atmosphere, airframe, n_t, objective)),
                                        promotes_inputs=[],
                                        promotes_outputs=[])

            # Create connections from trajectory group
            self.model.connect('trajectory.x', 'noise.x')
            self.model.connect('trajectory.y', 'noise.y')
            self.model.connect('trajectory.z', 'noise.z')
            self.model.connect('trajectory.alpha', 'noise.alpha')
            self.model.connect('trajectory.gamma', 'noise.gamma')
            self.model.connect('trajectory.t_s', 'noise.t_s')
            self.model.connect('trajectory.M_0', 'noise.M_0')

            if settings['core_jet_suppression'] and settings['case_name'] in ['nasa_stca_standard', 'stca_enginedesign_standard']:
                self.model.connect('trajectory.TS', 'noise.TS')
            
            if settings['atmosphere_type'] == 'stratified':
                self.model.connect('trajectory.c_0', 'noise.c_0')
                self.model.connect('trajectory.T_0', 'noise.T_0')
                self.model.connect('trajectory.rho_0', 'noise.rho_0')
                self.model.connect('trajectory.p_0', 'noise.p_0')
                self.model.connect('trajectory.mu_0', 'noise.mu_0')
                self.model.connect('trajectory.I_0', 'noise.I_0')

            # Create connections from engine component
            if settings['fan_inlet_source'] or settings['fan_discharge_source']:
                self.model.connect('engine.DTt_f', 'noise.DTt_f')
                self.model.connect('engine.mdot_f', 'noise.mdot_f')
                self.model.connect('engine.N_f', 'noise.N_f')
                self.model.connect('engine.A_f', 'noise.A_f')
                self.model.connect('engine.d_f', 'noise.d_f')
            if settings['core_source']:
                if settings['core_turbine_attenuation_method'] == 'ge':
                    self.model.connect('engine.mdoti_c', 'noise.mdoti_c')
                    self.model.connect('engine.Tti_c', 'noise.Tti_c')
                    self.model.connect('engine.Ttj_c', 'noise.Ttj_c')
                    self.model.connect('engine.Pti_c', 'noise.Pti_c')
                    self.model.connect('engine.DTt_des_c', 'noise.DTt_des_c')
                elif settings['core_turbine_attenuation_method'] == 'pw':
                    self.model.connect('engine.mdoti_c', 'noise.mdoti_c')
                    self.model.connect('engine.Tti_c', 'noise.Tti_c')
                    self.model.connect('engine.Ttj_c', 'noise.Ttj_c')
                    self.model.connect('engine.Pti_c', 'noise.Pti_c')
                    self.model.connect('engine.rho_te_c', 'noise.rho_te_c')
                    self.model.connect('engine.c_te_c', 'noise.c_te_c')
                    self.model.connect('engine.rho_ti_c', 'noise.rho_ti_c')
                    self.model.connect('engine.c_ti_c', 'noise.c_ti_c')
            if settings['jet_mixing_source'] and settings['jet_shock_source'] == False:
                self.model.connect('engine.V_j', 'noise.V_j')
                self.model.connect('engine.rho_j', 'noise.rho_j')
                self.model.connect('engine.A_j', 'noise.A_j')
                self.model.connect('engine.Tt_j', 'noise.Tt_j')
            elif settings['jet_shock_source'] and settings['jet_mixing_source'] == False:
                self.model.connect('engine.V_j', 'noise.V_j')
                self.model.connect('engine.A_j', 'noise.A_j')
                self.model.connect('engine.Tt_j', 'noise.Tt_j')
                self.model.connect('engine.M_j', 'noise.M_j')
            elif settings['jet_shock_source'] and settings['jet_mixing_source']:
                self.model.connect('engine.V_j', 'noise.V_j')
                self.model.connect('engine.rho_j', 'noise.rho_j')
                self.model.connect('engine.A_j', 'noise.A_j')
                self.model.connect('engine.Tt_j', 'noise.Tt_j')
                self.model.connect('engine.M_j', 'noise.M_j')
            if settings['airframe_source']:
                self.model.connect('trajectory.theta_flaps', 'noise.theta_flaps')
                self.model.connect('trajectory.I_landing_gear', 'noise.I_landing_gear')

        return None

    def set_objective(self, objective, noise_constraint_lateral=None):
        
        """
        """
        
        # Add objective for trajectory model
        if objective == None:
            pass 

        elif objective == 'x_end':
            self.model.add_objective('trajectory.x', index=-1, ref=1000.)
        
        elif objective == 'z_end':
            self.model.add_objective('trajectory.z', index=-1, ref=-1)

        elif objective == 't_end':
            self.model.add_objective('trajectory.t_s', index=-1, ref=1000.)
        
        elif objective == 'noise':
            self.model.add_constraint('noise.lateral', ref=1., upper=noise_constraint_lateral, units=None)
            self.model.add_objective('noise.flyover', ref=1.)

        else: 
            raise ValueError('Invalid optimization objective specified.')

        return None

    def set_phases_initial_conditions(self, airframe, z_cb, v_max, initialization_trajectory=None, trajectory_mode='cutback') -> None:
        
        # Set initial guess for the trajectory problem
        if initialization_trajectory is None:

            # Phase 1: groundroll
            if 'groundroll' in self.phase_name_lst:
                self['phases.groundroll.t_initial'] = 0.0
                self['phases.groundroll.t_duration'] = 30.0
                self['phases.groundroll.states:x'] = self.groundroll.interp(ys=[0, 1000], nodes='state_input')
                self['phases.groundroll.states:v'] = self.groundroll.interp(ys=[0.0, 60], nodes='state_input')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                self['phases.rotation.t_initial'] = 30.0
                self['phases.rotation.t_duration'] = 10.0
                self['phases.rotation.states:x'] = self.rotation.interp(ys=[1500, 2000], nodes='state_input')
                self['phases.rotation.states:v'] = self.rotation.interp(ys=[100, 110.], nodes='state_input')
                self['phases.rotation.states:alpha'] = self.rotation.interp(ys=[airframe.alpha_0, 15.], nodes='state_input')

            # Phase 3: lift-off
            if 'liftoff' in self.phase_name_lst:
                if trajectory_mode == 'flyover':
                    z_cutback_guess = 500.
                elif trajectory_mode == 'cutback':
                    z_cutback_guess = z_cb

                self['phases.liftoff.t_initial'] = 40.0
                self['phases.liftoff.t_duration'] = 2.
                self['phases.liftoff.states:x'] = self.liftoff.interp(ys=[2000., 3500.], nodes='state_input')
                self['phases.liftoff.states:z'] = self.liftoff.interp(ys=[0., 35*0.3048], nodes='state_input')
                self['phases.liftoff.states:v'] = self.liftoff.interp(ys=[110., v_max], nodes='state_input')
                self['phases.liftoff.states:gamma'] = self.liftoff.interp(ys=[0, 4.], nodes='state_input')
                self['phases.liftoff.controls:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='control_input')
                # self['phases.liftoff.states:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='state_input')

            # # Phase 4: vnrs 
            if 'vnrs' in self.phase_name_lst:
                self['phases.vnrs.t_initial'] = 50.0
                self['phases.vnrs.t_duration'] = 50.0
                self['phases.vnrs.states:x'] = self.vnrs.interp(ys=[3500., 6501.], nodes='state_input')
                self['phases.vnrs.states:z'] = self.vnrs.interp(ys=[35*0.3048, z_cutback_guess], nodes='state_input')
                self['phases.vnrs.states:v'] = self.vnrs.interp(ys=[v_max, v_max], nodes='state_input')
                self['phases.vnrs.states:gamma'] = self.vnrs.interp(ys=[4., 15.], nodes='state_input')
                self['phases.vnrs.controls:alpha'] = self.vnrs.interp(ys=[15., 15.], nodes='control_input')
                
            # Phase 5: cutback
            if 'cutback' in self.phase_name_lst:
                self['phases.cutback.t_initial'] = 100.0
                self['phases.cutback.t_duration'] = 50.0
                self['phases.cutback.states:x'] = self.cutback.interp(ys=[6501., 15000.], nodes='state_input')
                self['phases.cutback.states:z'] = self.cutback.interp(ys=[z_cutback_guess, 1500.], nodes='state_input')
                self['phases.cutback.states:v'] = self.cutback.interp(ys=[v_max, v_max], nodes='state_input')
                self['phases.cutback.states:gamma'] = self.cutback.interp(ys=[15, 15.], nodes='state_input')
                self['phases.cutback.controls:alpha'] = self.cutback.interp(ys=[15., 15.], nodes='control_input')
                
        else:

            # Phase 1: groundroll 
            if 'groundroll' in self.phase_name_lst:
                self['phases.groundroll.t_initial'] = initialization_trajectory.get_val('phases.groundroll.t_initial')
                self['phases.groundroll.t_duration'] = initialization_trajectory.get_val('phases.groundroll.t_duration')
                self['phases.groundroll.timeseries.time'] = initialization_trajectory.get_val('phases.groundroll.timeseries.time')
                self['phases.groundroll.states:x'] = initialization_trajectory.get_val('phases.groundroll.states:x')
                self['phases.groundroll.states:v'] = initialization_trajectory.get_val('phases.groundroll.states:v')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                self['phases.rotation.t_initial'] = initialization_trajectory.get_val('phases.rotation.t_initial')
                self['phases.rotation.t_duration'] = initialization_trajectory.get_val('phases.rotation.t_duration')
                self['phases.rotation.timeseries.time'] = initialization_trajectory.get_val('phases.rotation.timeseries.time')
                self['phases.rotation.states:x'] = initialization_trajectory.get_val('phases.rotation.states:x')
                self['phases.rotation.states:v'] = initialization_trajectory.get_val('phases.rotation.states:v')
                self['phases.rotation.states:alpha'] = initialization_trajectory.get_val('phases.rotation.states:alpha')

            # Phase 3-5: liftoff-cutback
            for _, phase_name in enumerate(self.phase_name_lst[2:]):
                self['phases.' + phase_name + '.t_initial'] = initialization_trajectory.get_val('phases.' + phase_name + '.t_initial')
                self['phases.' + phase_name + '.t_duration'] = initialization_trajectory.get_val('phases.' + phase_name + '.t_duration')
                self['phases.' + phase_name + '.timeseries.time'] = initialization_trajectory.get_val('phases.' + phase_name + '.timeseries.time')
                self['phases.' + phase_name + '.states:x'] = initialization_trajectory.get_val('phases.' + phase_name + '.states:x')
                self['phases.' + phase_name + '.states:z'] = initialization_trajectory.get_val('phases.' + phase_name + '.states:z')
                self['phases.' + phase_name + '.states:v'] = initialization_trajectory.get_val('phases.' + phase_name + '.states:v')
                self['phases.' + phase_name + '.states:gamma'] = initialization_trajectory.get_val('phases.' + phase_name + '.states:gamma')
                self['phases.' + phase_name + '.controls:alpha'] = initialization_trajectory.get_val('phases.' + phase_name + '.controls:alpha')

        return None

    def solve(self, run_driver, save_results) -> None:

        """
        """

        # Attach a recorder to the problem to save model data
        if save_results:
            self.add_recorder(om.SqliteRecorder(self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + self.output_file_name))

        # Run problem
        dm.run_problem(self, run_driver=run_driver)

        # Save the results
        if save_results:
            self.record(case_name=self.case_name)

        return None

    def simulate(self) -> om.Problem:

        """
        """

        return self.traj.simulate(times_per_seg=50)

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

    def plot(self, *problem_verify):

        # Check if problem_verify is empty
        if problem_verify:
            verification = True
            problem_verify = problem_verify[0]
        else:
            verification = False
        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use(self.pyna_directory + '/utils/' + 'plot.mplstyle')

        ax[0,0].plot(self.get_val('trajectory.x'), self.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        if verification:
            ax[0,0].plot(problem_verify['X [m]'], problem_verify['Z [m]'], '--', label='NASA STCA (Berton et al.)', color='tab:orange')
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.v'), '-', color='k')
        if verification:
            ax[0,1].plot(problem_verify['t_source [s]'], problem_verify['V [m/s]'], '--', label='NASA STCA (Berton et al.)', color='tab:orange')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.gamma'), '-', color='k')
        if verification:
            ax[0,2].plot(problem_verify['t_source [s]'], problem_verify['gamma [deg]'], '--', color='tab:orange')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.get_val('trajectory.t_s'), 1 / 1000. * self.get_val('engine.F_n'), '-', color='k')
        if verification:
            ax[1,0].plot(problem_verify['t_source [s]'], 1 / 1000. * problem_verify['F_n [N]'], '--', color='tab:orange')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.TS'), '-', color='k')
        if verification:
            ax[1,1].plot(problem_verify['t_source [s]'], problem_verify['TS [-]'], '--', color='tab:orange')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$TS$ [-]')
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.alpha'), '-', color='k')
        if verification:
            ax[1,2].plot(problem_verify['t_source [s]'], problem_verify['alpha [deg]'], '--', color='tab:orange')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None


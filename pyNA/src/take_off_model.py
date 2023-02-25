import openmdao.api as om
import dymos as dm
import pyNA

from pyNA.src.trajectory_model.take_off_phase_ode import TakeOffPhaseODE
from pyNA.src.trajectory_model.mux import Mux

from pyNA.src.trajectory_model.phases.groundroll import GroundRoll
from pyNA.src.trajectory_model.phases.rotation import Rotation
from pyNA.src.trajectory_model.phases.liftoff import LiftOff
from pyNA.src.trajectory_model.phases.vnrs import Vnrs
from pyNA.src.trajectory_model.phases.cutback import CutBack


class TakeOffModel(om.Problem):
    
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

    def get_var(self) -> None:

        self.vars.append('x'); self.var_units['x'] = 'm'
        self.vars.append('y'); self.var_units['y'] = 'm'
        self.vars.append('z'); self.var_units['z'] = 'm'
        self.vars.append('v'); self.var_units['v'] = 'm/s'
        self.vars.append('alpha'); self.var_units['alpha'] = 'deg'
        self.vars.append('gamma'); self.var_units['gamma'] = 'deg'
        self.vars.append('t_s'); self.var_units['t_s'] = 's'
        self.vars.append('M_0'); self.var_units['M_0'] = None
        self.vars.append('tau'); self.var_units['tau'] = None
        self.vars.append('c_0'); self.var_units['c_0'] = 'm/s'
        self.vars.append('T_0'); self.var_units['T_0'] = 'K'
        self.vars.append('rho_0'); self.var_units['rho_0'] = 'kg/m**3'
        self.vars.append('p_0'); self.var_units['p_0'] = 'Pa'
        self.vars.append('mu_0'); self.var_units['mu_0'] = 'kg/m/s'
        self.vars.append('I_0'); self.var_units['I_0'] = 'kg/m**2/s'
        self.vars.append('I_landing_gear'); self.var_units['I_landing_gear'] = None
        self.vars.append('theta_flaps'); self.var_units['theta_flaps'] = 'deg'
        self.vars.append('theta_slats'); self.var_units['theta_slats'] = 'deg'
        self.vars.append('c_l'); self.var_units['c_l'] = None
        self.vars.append('c_d'); self.var_units['c_d'] = None
        self.vars.append('c_l_max'); self.var_units['c_l_max'] = None
        self.vars.append('n'); self.var_units['n'] = None
        self.vars.append('mdot_NOx'); self.var_units['mdot_NOx'] = 'kg/s'
        self.vars.append('EINOx'); self.var_units['EINOx'] = None

        return None

    def create(self, settings, aircraft, trajectory_mode, objective):
    
        self.traj = dm.Trajectory()
        self.path.add_subsystem('phases', self.traj)

        # Create the trajectory phases 
        for phase_name in self.phase_name_lst:
            opts = {'phase': phase_name, 'settings': settings, 'aircraft': aircraft, 'objective': objective}
            
            if phase_name == 'groundroll':
                self.groundroll = GroundRoll(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.groundroll.create(settings, aircraft, objective)
                self.traj.add_phase(phase_name, self.groundroll)
            
            elif phase_name == 'rotation':
                self.rotation = Rotation(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.rotation.create(settings, aircraft, objective)
                self.traj.add_phase(phase_name, self.rotation)

            elif phase_name == 'liftoff':
                self.liftoff = LiftOff(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.liftoff.create(settings, aircraft, objective)
                self.traj.add_phase(phase_name, self.liftoff)

            elif phase_name == 'vnrs':
                self.vnrs = Vnrs(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.vnrs.create(settings, aircraft, objective, trajectory_mode)
                self.traj.add_phase(phase_name, self.vnrs)
                
            elif phase_name == 'cutback':
                self.cutback = CutBack(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.cutback.create(settings, aircraft, objective, trajectory_mode)
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
                self.traj.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='tau', var_b='tau', loc_a='final', loc_b='initial')

        if 'cutback' in self.phase_name_lst:
            if trajectory_mode == 'flyover':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'z', 'v', 'alpha', 'gamma'])
            elif trajectory_mode == 'cutback':
                self.traj.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['ptcb']:
                self.traj.add_linkage_constraint(phase_a='vnrs', phase_b='cutback', var_a='tau', var_b='tau', loc_a='final', loc_b='initial')

        # Mux trajectory and engine variables
        TakeOffModel.get_mux_input_output_size(self)
        mux_t = self.model.add_subsystem(name='trajectory', subsys=Mux(objective=objective, input_size_array=self.mux_input_size_array, output_size=self.mux_output_size, case_name=self.case_name, output_directory_name=self.output_directory_name))
        TakeOffModel.get_var()
        for var in self.var:
            
            mux_t.add_var('t_s', units=self.var_units[var])
            

            for j, phase_name in enumerate(self.phase_name_lst):
                if var == 't_s':
                    self.model.connect('phases.' + phase_name + '.interpolated.time', 'trajectory.' + var + '_' + str(j))

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

                elif var in ['tau']:
                    if objective == 'noise' and settings['ptcb']:
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

                elif var in ['eas', 'n','M_0', 'p_0','rho_0', 'T_0', 'c_0', 'c_bar', 'mu_0', 'I_0', 'mdot_NOx', 'EINOx', 'c_l', 'c_d', 'c_l_max']:
                    self.model.connect('phases.' + phase_name + '.interpolated.' + var, 'trajectory.' + var + '_' + str(j))

        mux_e = self.model.add_subsystem(name='engine', subsys=Mux(objective=objective, input_size_array=self.mux_input_size_array, output_size=self.mux_output_size, case_name=self.case_name, output_directory_name=self.output_directory_name))
        for var in aircraft.engine.vars:
            mux_e.add_var(var, units=aircraft.engine.var_units[var])
            for j, phase_name in enumerate(self.phase_name_lst):
                self.model.connect('phases.' + phase_name + '.interpolated.' + var, 'engine.' + var + '_' + str(j))

        return 

    def set_objective(self, objective) -> None:
        
        """
        """
        
        if objective == 'distance':
            self.model.add_objective('trajectory.x', index=-1, ref=1000.)

        elif objective == 'time':
            self.model.add_objective('trajectory.t_s', index=-1, ref=1000.)

        else:
            raise ValueError('Invalid optimization objective specified.')

        return None

    def set_driver_settings(self, settings, objective) -> None:
        
        """
        """

        # Set solver settings for the problem
        self.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        self.driver.opt_settings['print_level'] = 5
        self.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

        self.driver.declare_coloring(tol=1e-12)
        self.model.linear_solver = om.LinearRunOnce()
        self.driver.opt_settings['output_file'] = pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/IPOPT_trajectory_convergence.out'

        if objective == 'noise':
            self.driver.opt_settings['tol'] = 1e-3
            self.driver.opt_settings['acceptable_tol'] = 1e-1
        else:
            self.driver.opt_settings['tol'] = settings['tolerance']
            self.driver.opt_settings['acceptable_tol'] = 1e-2

        self.driver.opt_settings['max_iter'] = settings['max_iter']
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

    def set_initial_conditions(self, settings, aircraft, trajectory_mode, path_init=None) -> None:
        
        # Set initial guess for the trajectory problem
        if path_init is None:

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
                self['phases.rotation.states:alpha'] = self.rotation.interp(ys=[aircraft.alpha_0, 15.], nodes='state_input')

            # Phase 3: lift-off
            if 'liftoff' in self.phase_name_lst:
                if trajectory_mode == 'flyover':
                    z_cutback_guess = 500.
                elif trajectory_mode == 'cutback':
                    z_cutback_guess = settings['z_cb']

                self['phases.liftoff.t_initial'] = 40.0
                self['phases.liftoff.t_duration'] = 2.
                self['phases.liftoff.states:x'] = self.liftoff.interp(ys=[2000., 3500.], nodes='state_input')
                self['phases.liftoff.states:z'] = self.liftoff.interp(ys=[0., 35*0.3048], nodes='state_input')
                self['phases.liftoff.states:v'] = self.liftoff.interp(ys=[110., settings['v_max']], nodes='state_input')
                self['phases.liftoff.states:gamma'] = self.liftoff.interp(ys=[0, 4.], nodes='state_input')
                self['phases.liftoff.controls:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='control_input')
                # self['phases.liftoff.states:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='state_input')

            # # Phase 4: vnrs 
            if 'vnrs' in self.phase_name_lst:
                self['phases.vnrs.t_initial'] = 50.0
                self['phases.vnrs.t_duration'] = 50.0
                self['phases.vnrs.states:x'] = self.vnrs.interp(ys=[3500., 6501.], nodes='state_input')
                self['phases.vnrs.states:z'] = self.vnrs.interp(ys=[35*0.3048, z_cutback_guess], nodes='state_input')
                self['phases.vnrs.states:v'] = self.vnrs.interp(ys=[settings['v_max'], settings['v_max']], nodes='state_input')
                self['phases.vnrs.states:gamma'] = self.vnrs.interp(ys=[4., 15.], nodes='state_input')
                self['phases.vnrs.controls:alpha'] = self.vnrs.interp(ys=[15., 15.], nodes='control_input')
                
            # Phase 5: cutback
            if 'cutback' in self.phase_name_lst:
                self['phases.cutback.t_initial'] = 100.0
                self['phases.cutback.t_duration'] = 50.0
                self['phases.cutback.states:x'] = self.cutback.interp(ys=[6501., 15000.], nodes='state_input')
                self['phases.cutback.states:z'] = self.cutback.interp(ys=[z_cutback_guess, 1500.], nodes='state_input')
                self['phases.cutback.states:v'] = self.cutback.interp(ys=[settings['v_max'], settings['v_max']], nodes='state_input')
                self['phases.cutback.states:gamma'] = self.cutback.interp(ys=[15, 15.], nodes='state_input')
                self['phases.cutback.controls:alpha'] = self.cutback.interp(ys=[15., 15.], nodes='control_input')
                
        else:

            # Phase 1: groundroll 
            if 'groundroll' in self.phase_name_lst:
                self['phases.groundroll.t_initial'] = path_init.get_val('phases.groundroll.t_initial')
                self['phases.groundroll.t_duration'] = path_init.get_val('phases.groundroll.t_duration')
                self['phases.groundroll.timeseries.time'] = path_init.get_val('phases.groundroll.timeseries.time')
                self['phases.groundroll.states:x'] = path_init.get_val('phases.groundroll.states:x')
                self['phases.groundroll.states:v'] = path_init.get_val('phases.groundroll.states:v')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                self['phases.rotation.t_initial'] = path_init.get_val('phases.rotation.t_initial')
                self['phases.rotation.t_duration'] = path_init.get_val('phases.rotation.t_duration')
                self['phases.rotation.timeseries.time'] = path_init.get_val('phases.rotation.timeseries.time')
                self['phases.rotation.states:x'] = path_init.get_val('phases.rotation.states:x')
                self['phases.rotation.states:v'] = path_init.get_val('phases.rotation.states:v')
                self['phases.rotation.states:alpha'] = path_init.get_val('phases.rotation.states:alpha')

            # Phase 3-5: liftoff-cutback
            for _, phase_name in enumerate(self.phase_name_lst[2:]):
                self['phases.' + phase_name + '.t_initial'] = path_init.get_val('phases.' + phase_name + '.t_initial')
                self['phases.' + phase_name + '.t_duration'] = path_init.get_val('phases.' + phase_name + '.t_duration')
                self['phases.' + phase_name + '.timeseries.time'] = path_init.get_val('phases.' + phase_name + '.timeseries.time')
                self['phases.' + phase_name + '.states:x'] = path_init.get_val('phases.' + phase_name + '.states:x')
                self['phases.' + phase_name + '.states:z'] = path_init.get_val('phases.' + phase_name + '.states:z')
                self['phases.' + phase_name + '.states:v'] = path_init.get_val('phases.' + phase_name + '.states:v')
                self['phases.' + phase_name + '.states:gamma'] = path_init.get_val('phases.' + phase_name + '.states:gamma')
                self['phases.' + phase_name + '.controls:alpha'] = path_init.get_val('phases.' + phase_name + '.controls:alpha')

        return None
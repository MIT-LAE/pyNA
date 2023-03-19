import openmdao.api as om
import dymos as dm
import pyNA
import pdb

from pyNA.src.trajectory_model.sst.take_off_phase_ode import TakeOffPhaseODE
from pyNA.src.trajectory_model.sst.phases.groundroll import GroundRoll
from pyNA.src.trajectory_model.sst.phases.rotation import Rotation
from pyNA.src.trajectory_model.sst.phases.liftoff import LiftOff
from pyNA.src.trajectory_model.sst.phases.vnrs import Vnrs
from pyNA.src.trajectory_model.sst.phases.cutback import CutBack
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_model.mux import Mux


class SSTTakeOffTrajectory(dm.Trajectory):

    """
    
    Parameters
    ----------
    settings : dict
        pyNA settings
    aircraft : Aircraft
        _
    mode : str
        'cutback' or 'flyover'
    objective : str
        _

    Attributes
    ---------

    """

    def __init__(self, settings, aircraft:Aircraft, controls: dict, mode:str, objective:str, **kwargs):
        
        """

        Parameters
        ----------
        settings :
            _
        aircraft : 
            _
        controls : 
            _
        mode : str
            'cutback' or 'flyover'
        objective : str
            _

        Arguments
        ---------
        phase_name_lst : list
            _
        groundroll : Groundroll
            _
        rotation : Rotation
            _
        liftoff : Liftoff
            _
        vnrs: Vnrs
            _
        cutback : GroundRoll
            _

        """
        
        super().__init__(**kwargs)

        self.vars = list()
        self.var_units = dict()
        self.objective = objective
        self.mode = mode

        # Create the trajectory phases 
        self.phase_name_lst = ['groundroll', 'rotation', 'liftoff', 'vnrs', 'cutback']
        SSTTakeOffTrajectory.set_transcription(self)
        controls['tau_min'] = SSTTakeOffTrajectory.get_minimum_thrust_setting(self)

        for phase_name in self.phase_name_lst:
            opts = {'phase': phase_name, 'settings': settings, 'aircraft': aircraft, 'objective': objective}
            
            if phase_name == 'groundroll':
                self.groundroll = GroundRoll(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.groundroll.create(settings=settings, aircraft=aircraft, controls=controls, objective=objective)
                self.add_phase(phase_name, self.groundroll)
            
            elif phase_name == 'rotation':
                self.rotation = Rotation(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.rotation.create(settings=settings, aircraft=aircraft, controls=controls, objective=objective)
                self.add_phase(phase_name, self.rotation)

            elif phase_name == 'liftoff':
                self.liftoff = LiftOff(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.liftoff.create(settings=settings, aircraft=aircraft, controls=controls, objective=objective)
                self.add_phase(phase_name, self.liftoff)

            elif phase_name == 'vnrs':
                self.vnrs = Vnrs(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.vnrs.create(settings=settings, aircraft=aircraft, controls=controls, objective=objective, mode=mode)
                self.add_phase(phase_name, self.vnrs)
                
            elif phase_name == 'cutback':
                self.cutback = CutBack(ode_class=TakeOffPhaseODE, transcription=self.transcription[phase_name]['grid'], ode_init_kwargs=opts)
                self.cutback.create(settings=settings, aircraft=aircraft, controls=controls, objective=objective, mode=mode)
                self.add_phase(phase_name, self.cutback)        

        # Link phases
        if 'rotation' in self.phase_name_lst:
            self.link_phases(phases=['groundroll', 'rotation'], vars=['time', 'x', 'v', 'alpha'])    
            if objective == 'noise' and settings['phld']:
                self.add_linkage_constraint(phase_a='groundroll', phase_b='rotation', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'liftoff' in self.phase_name_lst:
            self.link_phases(phases=['rotation', 'liftoff'], vars=['time', 'x', 'z', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['phld']:
                self.add_linkage_constraint(phase_a='rotation', phase_b='liftoff', var_a='theta_flaps', var_b='theta_flaps', loc_a='final', loc_b='initial')

        if 'vnrs' in self.phase_name_lst:
            self.link_phases(phases=['liftoff', 'vnrs'],  vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['ptcb']:
                self.add_linkage_constraint(phase_a='liftoff', phase_b='vnrs', var_a='tau', var_b='tau', loc_a='final', loc_b='initial')

        if 'cutback' in self.phase_name_lst:
            if mode == 'flyover':
                self.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'z', 'v', 'alpha', 'gamma'])
            elif mode == 'cutback':
                self.link_phases(phases=['vnrs', 'cutback'], vars=['time', 'x', 'v', 'alpha', 'gamma'])
            if objective == 'noise' and settings['ptcb']:
                self.add_linkage_constraint(phase_a='vnrs', phase_b='cutback', var_a='tau', var_b='tau', loc_a='final', loc_b='initial')

        return None

    def connect_to_model(self, problem: om.Problem, settings: dict, aircraft: Aircraft):

        """
        
        Parameters
        ----------
        problem : om.Problem
            _
        settings : dict
            pyna settings
        aircraft : Aircraft
            _

        """

        # Connect trajectory to model
        problem.model.add_subsystem(name='phases', subsys=self)

        # Mux trajectory and engine variables
        promote_lst = ['t_s', 'x', 'y', 'z', 'v', 'alpha', 'gamma', 'F_n', 'tau', 'M_0', 'c_0', 'T_0', 'p_0', 'rho_0', 'mu_0', 'I_0']
        if settings['noise']:
            if settings['airframe_source']:
                promote_lst.extend(['theta_flaps', 'I_lg'])
            if settings['jet_mixing_source'] and not settings['jet_shock_source']:
                promote_lst.extend(['jet_V', 'jet_rho', 'jet_A', 'jet_Tt'])
            elif not settings['jet_mixing_source'] and settings['jet_shock_source']:
                promote_lst.extend(['jet_V', 'jet_A', 'jet_Tt', 'jet_M'])
            elif settings['jet_mixing_source'] and settings['jet_shock_source']:
                promote_lst.extend(['jet_V', 'jet_rho', 'jet_A', 'jet_Tt', 'jet_M'])
            if settings['core_source']:
                if settings['core_turbine_attenuation_method'] == "ge":
                    promote_lst.extend(['core_mdot', 'core_Tt_i', 'core_Tt_j', 'core_Pt_i', 'turb_DTt_des'])
                if settings['core_turbine_attenuation_method'] == "pw":
                    promote_lst.extend(['core_mdot', 'core_Tt_i', 'core_Tt_j', 'core_Pt_i', 'turb_rho_i', 'turb_c_i', 'turb_rho_e', 'turb_c_e'])
            if settings['fan_inlet_source'] or settings['fan_discharge_source']:
                promote_lst.extend(['fan_DTt', 'fan_mdot', 'fan_N'])

        SSTTakeOffTrajectory.get_mux_input_output_size(self)
        mux = problem.model.add_subsystem(name='trajectory', 
                                         subsys=Mux(input_size_array=self.mux_input_size_array, output_size=self.mux_output_size, settings=settings, objective=self.objective),
                                         promotes_outputs=promote_lst)
        
        SSTTakeOffTrajectory.get_var(self, settings=settings)
        for var in self.vars:
            
            mux.add_var(var, units=self.var_units[var])
            
            for j, phase_name in enumerate(self.phase_name_lst):
                if var == 't_s':
                    problem.model.connect('phases.' + phase_name + '.interpolated.time', 'trajectory.' + var + '_' + str(j))

                elif var in ['x', 'v']:
                    problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['z', 'gamma']:
                    if phase_name in {'groundroll','rotation'}:
                        problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                    else:
                        problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['alpha']:
                    if phase_name in {'groundroll'}:
                        problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var,'trajectory.' + var + '_' + str(j))
                    elif phase_name in {'rotation'}:
                        problem.model.connect('phases.' + phase_name + '.interpolated.states:' + var,'trajectory.' + var + '_' + str(j))
                    else:
                        problem.model.connect('phases.' + phase_name + '.interpolated.controls:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['tau']:
                    if self.objective == 'noise' and settings['ptcb']:
                        if phase_name in ['groundroll', 'rotation', 'liftoff']:
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'vnrs':
                            problem.model.connect('phases.' + phase_name + '.interpolated.controls:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'cutback':
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                    else:
                        if phase_name in ['groundroll', 'rotation', 'liftoff', 'vnrs']:
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))
                        elif phase_name == 'cutback':
                            problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['theta_flaps', 'theta_slats', 'y', 'I_lg']:
                    problem.model.connect('phases.' + phase_name + '.interpolated.parameters:' + var, 'trajectory.' + var + '_' + str(j))

                elif var in ['eas', 'n','M_0', 'p_0','rho_0', 'T_0', 'c_0', 'c_bar', 'mu_0', 'I_0', 'mdot_NOx', 'EINOx', 'c_l', 'c_d', 'c_l_max']:
                    problem.model.connect('phases.' + phase_name + '.interpolated.' + var, 'trajectory.' + var + '_' + str(j))

        for var in aircraft.engine.vars:
            mux.add_var(var, units=aircraft.engine.var_units[var])
            for j, phase_name in enumerate(self.phase_name_lst):
                problem.model.connect('phases.' + phase_name + '.interpolated.' + var, 'trajectory.' + var + '_' + str(j))

        return 

    def get_minimum_thrust_setting(self) -> float:

        tau_min = 0.6

        # TO DO: add routine here

        return tau_min

    def set_transcription(self) -> None:

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

        return None

    def set_initial_conditions(self, problem: om.Problem, settings: dict, aircraft:Aircraft, path_init=None) -> None:
        
        """
        
        Parameters
        ----------
        settings : dict
            -
        aircraft : Aircraft
            -
        path_init : 
            _

        """
        
        # Set initial guess for the trajectory problem
        if not path_init:

            # Phase 1: groundroll
            if 'groundroll' in self.phase_name_lst:
                problem['phases.groundroll.t_initial'] = 0.0
                problem['phases.groundroll.t_duration'] = 30.0
                problem['phases.groundroll.states:x'] = self.groundroll.interp(ys=[0, 1000], nodes='state_input')
                problem['phases.groundroll.states:v'] = self.groundroll.interp(ys=[0.0, 60], nodes='state_input')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                problem['phases.rotation.t_initial'] = 30.0
                problem['phases.rotation.t_duration'] = 10.0
                problem['phases.rotation.states:x'] = self.rotation.interp(ys=[1500, 2000], nodes='state_input')
                problem['phases.rotation.states:v'] = self.rotation.interp(ys=[100, 110.], nodes='state_input')
                problem['phases.rotation.states:alpha'] = self.rotation.interp(ys=[aircraft.alpha_0, 15.], nodes='state_input')

            # Phase 3: lift-off
            if 'liftoff' in self.phase_name_lst:
                if self.mode == 'flyover':
                    z_cutback_guess = 500.
                elif self.mode == 'cutback':
                    z_cutback_guess = settings['z_cb']

                problem['phases.liftoff.t_initial'] = 40.0
                problem['phases.liftoff.t_duration'] = 2.
                problem['phases.liftoff.states:x'] = self.liftoff.interp(ys=[2000., 3500.], nodes='state_input')
                problem['phases.liftoff.states:z'] = self.liftoff.interp(ys=[0., 35*0.3048], nodes='state_input')
                problem['phases.liftoff.states:v'] = self.liftoff.interp(ys=[110., settings['v_max']], nodes='state_input')
                problem['phases.liftoff.states:gamma'] = self.liftoff.interp(ys=[0, 4.], nodes='state_input')
                problem['phases.liftoff.controls:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='control_input')
                # problem['phases.liftoff.states:alpha'] = self.liftoff.interp(ys=[15., 15.], nodes='state_input')

            # # Phase 4: vnrs 
            if 'vnrs' in self.phase_name_lst:
                problem['phases.vnrs.t_initial'] = 50.0
                problem['phases.vnrs.t_duration'] = 50.0
                problem['phases.vnrs.states:x'] = self.vnrs.interp(ys=[3500., 6501.], nodes='state_input')
                problem['phases.vnrs.states:z'] = self.vnrs.interp(ys=[35*0.3048, z_cutback_guess], nodes='state_input')
                problem['phases.vnrs.states:v'] = self.vnrs.interp(ys=[settings['v_max'], settings['v_max']], nodes='state_input')
                problem['phases.vnrs.states:gamma'] = self.vnrs.interp(ys=[4., 15.], nodes='state_input')
                problem['phases.vnrs.controls:alpha'] = self.vnrs.interp(ys=[15., 15.], nodes='control_input')
                
            # Phase 5: cutback
            if 'cutback' in self.phase_name_lst:
                problem['phases.cutback.t_initial'] = 100.0
                problem['phases.cutback.t_duration'] = 50.0
                problem['phases.cutback.states:x'] = self.cutback.interp(ys=[6501., settings['x_max']], nodes='state_input')
                problem['phases.cutback.states:z'] = self.cutback.interp(ys=[z_cutback_guess, 1500.], nodes='state_input')
                problem['phases.cutback.states:v'] = self.cutback.interp(ys=[settings['v_max'], settings['v_max']], nodes='state_input')
                problem['phases.cutback.states:gamma'] = self.cutback.interp(ys=[15, 15.], nodes='state_input')
                problem['phases.cutback.controls:alpha'] = self.cutback.interp(ys=[15., 15.], nodes='control_input')
                
        else:

            # Phase 1: groundroll 
            if 'groundroll' in self.phase_name_lst:
                problem['phases.groundroll.t_initial'] = path_init.get_val('phases.groundroll.t_initial')
                problem['phases.groundroll.t_duration'] = path_init.get_val('phases.groundroll.t_duration')
                problem['phases.groundroll.timeseries.time'] = path_init.get_val('phases.groundroll.timeseries.time')
                problem['phases.groundroll.states:x'] = path_init.get_val('phases.groundroll.states:x')
                problem['phases.groundroll.states:v'] = path_init.get_val('phases.groundroll.states:v')

            # Phase 2: rotation
            if 'rotation' in self.phase_name_lst:
                problem['phases.rotation.t_initial'] = path_init.get_val('phases.rotation.t_initial')
                problem['phases.rotation.t_duration'] = path_init.get_val('phases.rotation.t_duration')
                problem['phases.rotation.timeseries.time'] = path_init.get_val('phases.rotation.timeseries.time')
                problem['phases.rotation.states:x'] = path_init.get_val('phases.rotation.states:x')
                problem['phases.rotation.states:v'] = path_init.get_val('phases.rotation.states:v')
                problem['phases.rotation.states:alpha'] = path_init.get_val('phases.rotation.states:alpha')

            # Phase 3-5: liftoff-cutback
            for _, phase_name in enumerate(self.phase_name_lst[2:]):
                problem['phases.' + phase_name + '.t_initial'] = path_init.get_val('phases.' + phase_name + '.t_initial')
                problem['phases.' + phase_name + '.t_duration'] = path_init.get_val('phases.' + phase_name + '.t_duration')
                problem['phases.' + phase_name + '.timeseries.time'] = path_init.get_val('phases.' + phase_name + '.timeseries.time')
                problem['phases.' + phase_name + '.states:x'] = path_init.get_val('phases.' + phase_name + '.states:x')
                problem['phases.' + phase_name + '.states:z'] = path_init.get_val('phases.' + phase_name + '.states:z')
                problem['phases.' + phase_name + '.states:v'] = path_init.get_val('phases.' + phase_name + '.states:v')
                problem['phases.' + phase_name + '.states:gamma'] = path_init.get_val('phases.' + phase_name + '.states:gamma')
                problem['phases.' + phase_name + '.controls:alpha'] = path_init.get_val('phases.' + phase_name + '.controls:alpha')

        return None
    
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
    
    def get_var(self, settings) -> None:

        """
        """
        
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
        self.vars.append('I_lg'); self.var_units['I_lg'] = None
        self.vars.append('theta_flaps'); self.var_units['theta_flaps'] = 'deg'
        self.vars.append('theta_slats'); self.var_units['theta_slats'] = 'deg'
        self.vars.append('c_l'); self.var_units['c_l'] = None
        self.vars.append('c_d'); self.var_units['c_d'] = None
        self.vars.append('c_l_max'); self.var_units['c_l_max'] = None
        self.vars.append('n'); self.var_units['n'] = None
        if settings['emissions']:
            self.vars.append('mdot_NOx'); self.var_units['mdot_NOx'] = 'kg/s'
            self.vars.append('EINOx'); self.var_units['EINOx'] = None

        return None
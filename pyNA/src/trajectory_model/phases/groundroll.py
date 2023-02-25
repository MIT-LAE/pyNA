import dymos as dm
import numpy as np
import pdb


class GroundRoll(dm.Phase):

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase, **kwargs)
        self.phase_size = int(self.options['transcription'].options['num_segments']*self.options['transcription'].options['order'] + 1)

        self.phase_target_size = 10

    def create(self, settings, aircraft, controls, objective) -> None:

        self.set_time_options(fix_initial=True, duration_bounds=(0, 100), duration_ref=100.)
        
        self.add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=True, fix_final=False, ref=1000.)
        self.add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=True, fix_final=False, ref=100.)
        
        self.add_parameter('alpha', targets='alpha', units='deg', dynamic=True, include_timeseries=True, val=aircraft.alpha_0)
        self.add_parameter('z', targets='z', units='m', val=0., dynamic=True,include_timeseries=True)
        self.add_parameter('gamma', targets='gamma', units='deg', val=0., dynamic=True, include_timeseries=True)
        self.add_parameter('tau', targets='propulsion.tau', units=None, val=controls['tau']['groundroll'], dynamic=True, include_timeseries=True)
        if settings['pkrot']:
            self.add_parameter('k_rot', targets='flight_dynamics.k_rot', units=None, lower=1.1, upper=2.0, dynamic=False, val=controls['k_rot'], opt=True)
        else:
            self.add_parameter('k_rot', targets='flight_dynamics.k_rot', units=None, dynamic=False, val=controls['k_rot'], opt=False)
        if objective == 'noise' and settings['phld']:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=controls['theta_flaps']['groundroll'], dynamic=True, include_timeseries=True, opt=True, ref=10)
        else:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=controls['theta_flaps']['groundroll'], dynamic=True, include_timeseries=True)
        self.add_parameter('theta_slats', targets='theta_slats', units='deg', val=controls['theta_slats']['groundroll'], dynamic=True, include_timeseries=True)
        self.add_parameter('I_lg', units=None, val=1, dynamic=True, include_timeseries=True)
        self.add_parameter('y', units='m', val=0, dynamic=True, include_timeseries=True)

        self.add_boundary_constraint('flight_dynamics.v_rot_residual', equals=0., loc='final', ref=100, units='m/s')

        self.add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_target_size-1,order=3, solve_segments=False, compressed=True), subset='state_input')
        for var in aircraft.engine.vars: 
            self.add_timeseries_output('propulsion.'+ var, timeseries='interpolated')
        
        self.add_timeseries_output('p_0', timeseries='interpolated')
        self.add_timeseries_output('rho_0', timeseries='interpolated')
        self.add_timeseries_output('I_0', timeseries='interpolated')
        self.add_timeseries_output('drho_0_dz', timeseries='interpolated')
        self.add_timeseries_output('T_0', timeseries='interpolated')
        self.add_timeseries_output('c_0', timeseries='interpolated')
        self.add_timeseries_output('mu_0', timeseries='interpolated')
        self.add_timeseries_output('flight_dynamics.n', timeseries='interpolated')
        self.add_timeseries_output('flight_dynamics.M_0', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_l', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_l_max', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_d', timeseries='interpolated')
        if settings['emissions']:
            self.add_timeseries_output('emissions.mdot_NOx', timeseries='interpolated')
            self.add_timeseries_output('emissions.EINOx', timeseries='interpolated')
            
        return None
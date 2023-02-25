import dymos as dm
import numpy as np


class Rotation(dm.Phase):

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase, **kwargs)
        self.phase_size = int(self.options['transcription'].options['num_segments']*self.options['transcription'].options['order'] + 1)

        self.phase_target_size = 10

    def create(self, settings, airframe, engine, objective) -> None:

        self.set_time_options(initial_bounds=(10, 100), duration_bounds=(0, 100), initial_ref=100., duration_ref=100.)
        
        self.add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=1000.)
        self.add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
        self.add_state('alpha', targets='alpha', rate_source='flight_dynamics.alpha_dot', units='deg', fix_initial=False, fix_final=False, lower=airframe.aero['alpha'][0], upper=airframe.aero['alpha'][-1], ref=10.)
        
        self.add_parameter('z', targets='z', units='m', val=0., dynamic=True,include_timeseries=True)
        self.add_parameter('gamma', targets='gamma', units='deg', val=0., dynamic=True, include_timeseries=True)
        self.add_parameter('tau', targets='propulsion.tau', units=None, val=tau, dynamic=True, include_timeseries=True)
        if objective == 'noise' and settings['phld']:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=theta_flaps, dynamic=True, include_timeseries=True, opt=True, ref=10.)
        else:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=theta_flaps, dynamic=True, include_timeseries=True)
        self.add_parameter('theta_slats', targets='theta_slats', units='deg', val=theta_slats, dynamic=True, include_timeseries=True)
        self.add_parameter('I_landing_gear', units=None, val=1, dynamic=True, include_timeseries=True)
        self.add_parameter('y', units='m', val=0, dynamic=True, include_timeseries=True)

        self.add_boundary_constraint('flight_dynamics.n', equals=1.1, loc='final', ref=1, units=None)
        
        self.add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_target_size-1, order=3, solve_segments=False, compressed=True), subset='state_input')
        for var in engine.deck_variables.keys():
            self.add_timeseries_output('propulsion.'+ var, timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.M_0', timeseries='interpolated')
        self.add_timeseries_output('p_0', timeseries='interpolated')
        self.add_timeseries_output('rho_0', timeseries='interpolated')
        self.add_timeseries_output('I_0', timeseries='interpolated')
        self.add_timeseries_output('drho_0_dz', timeseries='interpolated')
        self.add_timeseries_output('T_0', timeseries='interpolated')
        self.add_timeseries_output('c_0', timeseries='interpolated')
        self.add_timeseries_output('mu_0', timeseries='interpolated')
        self.add_timeseries_output('emissions.mdot_NOx', timeseries='interpolated')
        self.add_timeseries_output('emissions.EINOx', timeseries='interpolated')
        self.add_timeseries_output('flight_dynamics.n', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.L', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.D', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_l', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_l_max', timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.c_d', timeseries='interpolated')

        return None

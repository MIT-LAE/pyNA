import dymos as dm
import numpy as np


class Vnrs(dm.Phase):

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase, **kwargs)
        self.phase_size = int(self.options['transcription'].options['num_segments']*self.options['transcription'].options['order'] + 1)

        self.phase_target_size = 22

    def create(self, airframe, engine, ptcb, phld, TS_vnrs, TS_min, theta_flaps, theta_slats, trajectory_mode, objective, atmosphere_type) -> None:
        
        self.set_time_options(initial_bounds=(10, 300), duration_bounds=(0, 500), initial_ref=100., duration_ref=100.)            
        
        if trajectory_mode == 'flyover':
            self.add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=True, ref=10000.)
            self.add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=True, fix_final=False, ref=1000.)
        elif trajectory_mode == 'cutback':
            self.add_state('x', rate_source='flight_dynamics.x_dot', units='m', fix_initial=False, fix_final=False, ref=10000.)
            self.add_state('z', rate_source='flight_dynamics.z_dot', units='m', fix_initial=True, fix_final=True, ref=1000.)
        self.add_state('v', targets='v', rate_source='flight_dynamics.v_dot', units='m/s', fix_initial=False, fix_final=False, ref=100.)
        self.add_state('gamma', rate_source='flight_dynamics.gamma_dot', units='deg', fix_initial=False, fix_final=False, ref=10.)
        self.add_control('alpha', targets='alpha', units='deg', lower=5., upper=airframe.aero['alpha'][-1], rate_continuity=True, rate_continuity_scaler=1.0, rate2_continuity=False, opt=True, ref=10.)
        
        if objective == 'noise' and ptcb:
            self.add_control('TS', targets='propulsion.TS', units=None, upper=1, lower=TS_min, val=TS_vnrs, opt=True, rate_continuity=True, rate2_continuity=False, ref=1.)
        else:
            self.add_parameter('TS', targets='propulsion.TS', units=None, val=TS_vnrs, dynamic=True, include_timeseries=True)
    
        if objective == 'noise' and phld:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=0., dynamic=True, include_timeseries=True, ref=10.)
        else:
            self.add_parameter('theta_flaps', targets='theta_flaps', units='deg', val=theta_flaps, dynamic=True, include_timeseries=True)
        self.add_parameter('theta_slats', targets='theta_slats', units='deg', val=theta_slats, dynamic=True, include_timeseries=True)
        self.add_parameter('I_landing_gear', units=None, val=0, dynamic=True, include_timeseries=True)
        self.add_parameter('y', units='m', val=0, dynamic=True, include_timeseries=True)

        self.add_path_constraint(name='flight_dynamics.v_dot', lower=0., units='m/s**2')
        self.add_path_constraint(name='gamma', lower=0., units='deg', ref=10.)

        self.add_timeseries('interpolated', transcription=dm.GaussLobatto(num_segments=self.phase_target_size-1, order=3, solve_segments=False, compressed=True), subset='state_input')
        for var in engine.deck_variables.keys():
            self.add_timeseries_output('propulsion.'+ var, timeseries='interpolated')
        self.add_timeseries_output('aerodynamics.M_0', timeseries='interpolated')
        if atmosphere_type == 'stratified':
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
        self.add_timeseries_output('clcd.c_l', timeseries='interpolated')
        self.add_timeseries_output('clcd.c_l_max', timeseries='interpolated')
        self.add_timeseries_output('clcd.c_d', timeseries='interpolated')
        
        return None
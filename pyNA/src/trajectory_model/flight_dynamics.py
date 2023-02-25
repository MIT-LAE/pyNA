import pdb
import openmdao
import datetime as dt
import numpy as np
from openmdao.api import ExplicitComponent
import pyNA
from pyNA.src.aircraft import Aircraft


class FlightDynamics(ExplicitComponent):
    """
    Computes flight dynamics parameters along the trajectory.

    The *FlightDynamics* component requires the following inputs:

    * ``inputs['x']``:                  aircraft x-position [m]
    * ``inputs['z']``:                  aircraft z-position [m]
    * ``inputs['v']``:                  aircraft velocity [m/s]
    * ``inputs['alpha']``:              aircraft angle of attack [deg]
    * ``inputs['gamma']``:              aircraft climb angle [deg]
    * ``inputs['F_n']``:                aircraft net thrust [N]
    * ``inputs['c_l']``:                aircraft lift coefficient [-]
    * ``inputs['c_d']``:                aircraft drag coefficient [-]

    The *FlightDynamics* component computes the following outputs:

    * ``outputs['x_dot']``:             rate of change of aircraft x-position [m/s]
    * ``outputs['z_dot']``:             rate of change of aircraft z-position [m/s]
    * ``outputs['alpha_dot']``:         rate of change of aircraft angle of attack [deg/s]
    * ``outputs['gamma_dot']``:         rate of change of aircraft climb angle [deg/s]
    * ``outputs['v_dot']``:             rate of change of aircraft velocity [m/s2]
    * ``outputs['net_F_up']``:          aircraft net upward force [N]

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('phase', types=str)
        self.options.declare('settings', types=dict)
        self.options.declare('aircraft', types=Aircraft)
        self.options.declare('objective', types=str)
        self.options.declare('constant_LD', default=None)
        
        self.g = 9.80665

    def setup(self):
        nn = self.options['num_nodes']
        phase_name = self.options['phase']

        # inputs
        self.add_input(name='x', val=np.ones(nn), desc='position along the trajectory', units='m')
        self.add_input(name='z', val=np.ones(nn), desc='altitude', units='m')
        self.add_input(name='v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input(name='alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input(name='gamma', val=np.ones(nn), desc='flight path angle', units='deg')
        self.add_input(name='F_n', val=np.ones(nn), desc='thrust', units='N')
        self.add_input(name='c_0', val=np.ones(nn), desc='ambient speed of sound', units='m/s')
        self.add_input(name='rho_0', val=np.ones(nn), desc='ambient density', units='kg/m**3')
        self.add_input(name='c_l', val=np.ones(nn), desc='lift coefficient', units=None)
        self.add_input(name='c_d', val=np.ones(nn), desc='drag coefficient', units=None)
        self.add_input(name='c_l_max', val=np.ones(nn), desc='maximum aircraft lift coefficient', units=None)
        if phase_name == 'groundroll':
            self.add_input(name='k_rot', val=1.2, desc='rotational speed parameter', units=None)

        # Outputs
        self.add_output(name='x_dot', val=np.ones(nn), desc='Position rate along the ground', units='m/s')
        self.add_output(name='z_dot', val=np.ones(nn), desc='rate of change of altitude', units='m/s')
        self.add_output(name='v_dot', val=np.ones(nn), desc='Acceleration along the ground (assuming zero wind)', units='m/s**2')        
        self.add_output(name='alpha_dot', val=np.ones(nn), desc='rate of change of angle of attack', units='deg/s')
        self.add_output(name='gamma_dot', val=np.ones(nn), desc='rate of change of flight path angle', units='deg/s')
        self.add_output(name='M_0', val=np.ones(nn), desc='flight Mach number', units=None)
        self.add_output(name='n', val=np.ones(nn), desc='load factor', units=None)
        if phase_name == 'groundroll':
            self.add_output(name='v_rot_residual', val=np.ones(nn), desc='residual v - v_rot', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='x_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='x_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='z_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='z_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='c_l', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='c_d', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='rho_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='c_l', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='rho_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='M_0', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='M_0', wrt='c_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='c_l', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='rho_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='v', rows=arange, cols=arange, val=1.0)
        if phase_name == 'groundroll':
            self.declare_partials(of='v_rot_residual', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='c_l_max', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='rho_0', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='k_rot', val=1.0)
            

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        aircraft = self.options['aircraft']
        settings = self.options['settings']

        calpha = np.cos((aircraft.inc_F_n + inputs['alpha'] - aircraft.alpha_0)*np.pi/180.)
        salpha = np.sin((aircraft.inc_F_n + inputs['alpha'] - aircraft.alpha_0)*np.pi/180.)
        cgamma = np.cos(inputs['gamma']*np.pi/180.)
        sgamma = np.sin(inputs['gamma']*np.pi/180.)

        q = 0.5 * inputs['rho_0'] * inputs['v'] ** 2
        L = q * aircraft.af_S_w * inputs['c_l']
        if self.options['constant_LD']: 
            D = q * aircraft.af_S_w * inputs['c_l']/self.options['constant_LD']
        else:
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                D = q * aircraft.af_S_w * (inputs['c_d'] + aircraft.c_d_g)
            elif phase_name in {'vnrs', 'cutback'}:
                D = q * aircraft.af_S_w * inputs['c_d']

        # dx/dt, dz/dt
        outputs['x_dot'] = inputs['v'] * cgamma
        outputs['z_dot'] = inputs['v'] * sgamma
       
        # dv/dt
        if phase_name in {'groundroll', 'rotation'}:
            F_fric = aircraft.mu_r * (aircraft.mtow * self.g - L)
            outputs['v_dot'] = (1.0 / aircraft.mtow) * (aircraft.n_eng*inputs['F_n'] * calpha - D - F_fric) - self.g * sgamma
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['v_dot'] = (1.0 / aircraft.mtow) * (aircraft.n_eng*inputs['F_n'] * calpha - D) - self.g * sgamma

        # dgamma/dt
        if phase_name in {'groundroll', 'rotation'}:
            outputs['gamma_dot'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['gamma_dot'] = ((aircraft.n_eng*inputs['F_n'] * salpha + L - aircraft.mtow * self.g * cgamma) / (aircraft.mtow * inputs['v']))*180/np.pi

        # Flight Mach number
        outputs['M_0'] = inputs['v'] / inputs['c_0']

        # Compute net upward force
        outputs['n'] = (aircraft.n_eng*inputs['F_n'] * salpha + L)/(aircraft.mtow * self.g * cgamma)

        # Stick (alpha) controller
        if phase_name in {'groundroll'}:
            outputs['alpha_dot'] = np.zeros(nn)
        elif phase_name in {'rotation'}:
            outputs['alpha_dot'] = 3.5 * np.ones(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['alpha_dot'] = np.zeros(nn)

        # Compute rotation speed residual
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * aircraft.mtow * self.g / (aircraft.af_S_w * inputs['rho_0'] * inputs['c_l_max']))
            v_rot = inputs['k_rot'] * v_stall
            outputs['v_rot_residual'] = inputs['v'] - v_rot

        # Print to file
        if self.options['objective'] == 'noise':
            # Write k to file
            if phase_name == 'groundroll':
                f = open(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/' + 'inputs_k.txt' , 'a')
                f.write(str(inputs['k_rot'][0]) + '\n')
                f.close()

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        aircraft =  self.options['aircraft']

        calpha = np.cos((aircraft.inc_F_n + inputs['alpha'] - aircraft.alpha_0)*np.pi/180.)
        c2alpha = np.cos(2*(aircraft.inc_F_n + inputs['alpha'] - aircraft.alpha_0)*np.pi/180.)
        salpha = np.sin((aircraft.inc_F_n + inputs['alpha'] - aircraft.alpha_0)*np.pi/180.)
        cgamma = np.cos(inputs['gamma']*np.pi/180.)
        c2gamma = np.cos(2*inputs['gamma']*np.pi/180.)
        sgamma = np.sin(inputs['gamma']*np.pi/180.)

        q = 0.5 * inputs['rho_0'] * inputs['v'] ** 2
        L = q * aircraft.af_S_w * inputs['c_l']
        if self.options['constant_LD']: 
            D = q * aircraft.af_S_w * inputs['c_l']/self.options['constant_LD']
        else:
            if phase_name in {'groundroll', 'rotation', 'liftoff'}:
                D = q * aircraft.af_S_w * (inputs['c_d'] + aircraft.c_d_g)
            elif phase_name in {'vnrs', 'cutback'}:
                D = q * aircraft.af_S_w * inputs['c_d']

        F_fric = aircraft.mu_r * (aircraft.mtow * self.g - L)
        xdot = inputs['v'] * cgamma
        zdot = inputs['v'] * sgamma
        
        # dx/dt, dz/dt
        partials['x_dot', 'gamma'] = -inputs['v'] * sgamma * np.pi / 180.
        partials['x_dot', 'v'] = cgamma

        partials['z_dot', 'gamma'] = inputs['v'] * cgamma * np.pi/180.
        partials['z_dot', 'v'] = sgamma

        # dv/dt
        if phase_name in {'groundroll', 'rotation'}:
            v_dot = (1.0 / aircraft.mtow) * (aircraft.n_eng*inputs['F_n'] * calpha - D - F_fric) - self.g * sgamma            
            partials['v_dot', 'F_n'] = calpha / aircraft.mtow * aircraft.n_eng
            partials['v_dot', 'alpha'] = -aircraft.n_eng*inputs['F_n'] * salpha / aircraft.mtow * np.pi/180.
            partials['v_dot', 'gamma'] = -self.g * cgamma * np.pi/180.
            if self.options['constant_LD']:
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * inputs['c_l']/self.options['constant_LD'] * 0.5*inputs['v']**2 * aircraft.af_S_w + aircraft.mu_r / aircraft.mtow * inputs['c_l'] * 0.5*inputs['v']**2* aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * inputs['c_l']/self.options['constant_LD'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w + aircraft.mu_r / aircraft.mtow * inputs['c_l'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w
                partials['v_dot', 'c_d'] = np.zeros(nn,)
                partials['v_dot', 'c_l'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w / self.options['constant_LD'] + aircraft.mu_r / aircraft.mtow * q * aircraft.af_S_w
            else:
                partials['v_dot', 'c_d'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w
                partials['v_dot', 'c_l'] = aircraft.mu_r / aircraft.mtow * q * aircraft.af_S_w
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * (inputs['c_d'] + aircraft.c_d_g) * 0.5*inputs['v']**2 * aircraft.af_S_w + aircraft.mu_r / aircraft.mtow * inputs['c_l'] * 0.5*inputs['v']**2* aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * (inputs['c_d'] + aircraft.c_d_g) * inputs['rho_0']*inputs['v'] * aircraft.af_S_w + aircraft.mu_r / aircraft.mtow * inputs['c_l'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w

        elif phase_name in {'liftoff'}:
            v_dot = (1.0 / aircraft.mtow) * (aircraft.n_eng*inputs['F_n'] * calpha - D) - self.g * sgamma
            partials['v_dot', 'F_n'] = calpha / aircraft.mtow * aircraft.n_eng
            partials['v_dot', 'alpha'] = -aircraft.n_eng*inputs['F_n'] * salpha / aircraft.mtow * np.pi/180.
            partials['v_dot', 'gamma'] = -self.g * cgamma * np.pi/180.
            if self.options['constant_LD']:
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * inputs['c_l']/self.options['constant_LD'] * 0.5*inputs['v']**2 * aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * inputs['c_l']/ self.options['constant_LD'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w
                partials['v_dot', 'c_d'] = np.zeros(nn,)
                partials['v_dot', 'c_l'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w / self.options['constant_LD']
            else:
                partials['v_dot', 'c_d'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w
                partials['v_dot', 'c_l'] = np.zeros(nn,)
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * (inputs['c_d'] + aircraft.c_d_g) * 0.5*inputs['v']**2 * aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * (inputs['c_d'] + aircraft.c_d_g) * inputs['rho_0']*inputs['v'] * aircraft.af_S_w

        elif phase_name in {'vnrs', 'cutback'}:
            v_dot = (1.0 / aircraft.mtow) * (aircraft.n_eng*inputs['F_n'] * calpha - D) - self.g * sgamma
            partials['v_dot', 'F_n'] = calpha / aircraft.mtow * aircraft.n_eng
            partials['v_dot', 'alpha'] = -aircraft.n_eng*inputs['F_n'] * salpha / aircraft.mtow * np.pi/180.
            partials['v_dot', 'gamma'] = -self.g * cgamma * np.pi/180.
            if self.options['constant_LD']:
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * inputs['c_l']/self.options['constant_LD'] * 0.5*inputs['v']**2 * aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * inputs['c_l']/ self.options['constant_LD'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w
                partials['v_dot', 'c_d'] = np.zeros(nn,)
                partials['v_dot', 'c_l'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w / self.options['constant_LD']
            else:
                partials['v_dot', 'c_d'] = - 1.0 / aircraft.mtow * q * aircraft.af_S_w
                partials['v_dot', 'c_l'] = np.zeros(nn,)
                partials['v_dot', 'rho_0'] = - 1.0 / aircraft.mtow * inputs['c_d'] * 0.5*inputs['v']**2 * aircraft.af_S_w
                partials['v_dot', 'v'] = - 1.0 / aircraft.mtow * inputs['c_d'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w

        # Change in climb angle
        if phase_name in {'groundroll', 'rotation'}:
            gamma_dot = np.zeros(nn)
            partials['gamma_dot', 'F_n'] = np.zeros(nn)
            partials['gamma_dot', 'c_l'] = np.zeros(nn)
            partials['gamma_dot', 'gamma'] = np.zeros(nn)
            partials['gamma_dot', 'alpha'] = np.zeros(nn)
            partials['gamma_dot', 'v'] = np.zeros(nn)
            partials['gamma_dot', 'rho_0'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            gamma_dot = ((aircraft.n_eng*inputs['F_n'] * salpha + L - aircraft.mtow * self.g * cgamma) / (aircraft.mtow * inputs['v']))*180/np.pi
            partials['gamma_dot', 'F_n'] = (salpha / (aircraft.mtow * inputs['v']))*180/np.pi*aircraft.n_eng
            partials['gamma_dot', 'c_l'] = (1.0 / (aircraft.mtow * inputs['v'])) * q * aircraft.af_S_w * 180/np.pi
            partials['gamma_dot', 'rho_0'] = (1.0 / (aircraft.mtow * inputs['v'])) * inputs['c_l'] * 0.5 * inputs['v']**2 * aircraft.af_S_w * 180/np.pi
            partials['gamma_dot', 'gamma'] = (self.g * sgamma * np.pi/180. / inputs['v'])*180/np.pi
            partials['gamma_dot', 'alpha'] = (aircraft.n_eng*inputs['F_n'] * calpha * np.pi/180. / (aircraft.mtow * inputs['v']))*180/np.pi
            partials['gamma_dot', 'v'] = (self.g * cgamma / inputs['v'] ** 2 - aircraft.n_eng*inputs['F_n'] * salpha / (inputs['v'] ** 2 * aircraft.mtow) + inputs['c_l']*0.5*inputs['rho_0']*aircraft.af_S_w/aircraft.mtow)*180/np.pi

        # Mach number
        partials['M_0', 'v'] = 1/inputs['c_0']
        partials['M_0', 'c_0'] = -inputs['v']/inputs['c_0']**2

        # Net upward force
        partials['n', 'c_l'] = 1.0 / (aircraft.mtow * self.g * cgamma) * q * aircraft.af_S_w
        partials['n', 'rho_0'] = 1.0 / (aircraft.mtow * self.g * cgamma) * inputs['c_l'] * 0.5 * inputs['v']**2 * aircraft.af_S_w
        partials['n', 'F_n'] = salpha * aircraft.n_eng / (aircraft.mtow * self.g * cgamma)
        partials['n', 'alpha'] = aircraft.n_eng*inputs['F_n'] * calpha * (np.pi/180.) / (aircraft.mtow * self.g * cgamma)
        partials['n', 'gamma'] = (aircraft.n_eng*inputs['F_n'] * salpha + L) / (aircraft.mtow * self.g) / cgamma**2 * sgamma * (np.pi/180.)
        partials['n', 'v'] = 1.0 / (aircraft.mtow * self.g * cgamma) * inputs['c_l'] * inputs['rho_0']*inputs['v'] * aircraft.af_S_w

        # Compute rotation speed residual (v_residual = v - (aircraft.k_rot * v_stall))
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * aircraft.mtow * self.g / (aircraft.af_S_w * inputs['rho_0'] * inputs['c_l_max']))
            partials['v_rot_residual', 'v'] = np.ones(nn)
            partials['v_rot_residual', 'c_l_max'] = inputs['k_rot'] / 2 / v_stall * 2 * aircraft.mtow * self.g / (aircraft.af_S_w * inputs['rho_0'] * inputs['c_l_max'])**2 * (aircraft.af_S_w * inputs['rho_0'])
            partials['v_rot_residual', 'rho_0'] = inputs['k_rot'] / 2 / v_stall * 2 * aircraft.mtow * self.g / (aircraft.af_S_w * inputs['rho_0'] * inputs['c_l_max'])**2 * (aircraft.af_S_w * inputs['c_l_max'])
            partials['v_rot_residual', 'k_rot'] = - v_stall
            
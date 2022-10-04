import pdb
import openmdao
import datetime as dt
import numpy as np
from openmdao.api import ExplicitComponent
from pyNA.src.settings import Settings

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
    * ``inputs['L']``:                  aircraft lift [N]
    * ``inputs['D']``:                  aircraft drag [N]
    * ``inputs['rho_0']``:              ambient density [kg/m3]
    * ``inputs['c_0']``:                ambient speed of sound [m/s]
    * ``inputs['drho_0_dz']``:          change in atmospheric density with change in altitude [kg/m4]

    The *FlightDynamics* component computes the following outputs:

    * ``outputs['x_dot']``:             rate of change of aircraft x-position [m/s]
    * ``outputs['z_dot']``:             rate of change of aircraft z-position [m/s]
    * ``outputs['alpha_dot']``:         rate of change of aircraft angle of attack [deg/s]
    * ``outputs['gamma_dot']``:         rate of change of aircraft climb angle [deg/s]
    * ``outputs['v_dot']``:             rate of change of aircraft velocity [m/s2]
    * ``outputs['eas_dot']``:           rate of change of aircraft equivalent airspeed [m/s2]
    * ``outputs['y']``:                 aircraft lateral position [m]
    * ``outputs['eas']``:               aircraft equivalent airspeed [m/s]
    * ``outputs['net_F_up']``:          aircraft net upward force [N]
    * ``outputs['I_landing gear']``:    flag for landing gear extraction [-]

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('settings', types=Settings)
        self.options.declare('phase', types=str)
        self.options.declare('ac')
        self.options.declare('objective')

    def setup(self):
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        ac = self.options['ac']

        # inputs
        self.add_input(name='x', val=np.ones(nn), desc='position along the trajectory', units='m')
        self.add_input(name='z', val=np.ones(nn), desc='altitude', units='m')
        self.add_input(name='v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input(name='alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input(name='gamma', val=np.ones(nn), desc='flight path angle', units='deg')
        self.add_input(name='F_n', val=np.ones(nn), desc='thrust', units='N')
        self.add_input(name='L', val=np.ones(nn), desc='lift', units='N')
        self.add_input(name='D', val=np.ones(nn), desc='drag', units='N')
        self.add_input(name='rho_0', val=np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='c_0', val=np.ones(nn), desc='ambient speed of sound', units='m/s')
        self.add_input(name='drho_0_dz', val=np.ones(nn), desc='change in atmospheric density with change in altitude', units='kg/m**4')
        self.add_input(name='c_l_max', val=np.ones(nn), desc='maximum aircraft lift coefficient', units=None)
        if phase_name == 'groundroll':
            self.add_input(name='k_rot', val=ac.k_rot, desc='rotational speed parameter', units=None)

        # outputs
        self.add_output(name='x_dot', val=np.ones(nn), desc='Position rate along the ground', units='m/s')
        self.add_output(name='z_dot', val=np.ones(nn), desc='rate of change of altitude', units='m/s')
        self.add_output(name='alpha_dot', val=np.ones(nn), desc='rate of change of angle of attack', units='deg/s')
        self.add_output(name='gamma_dot', val=np.ones(nn), desc='rate of change of flight path angle', units='deg/s')
        self.add_output(name='v_dot', val=np.ones(nn), desc='Acceleration along the ground (assuming zero wind)', units='m/s**2')
        self.add_output(name='eas_dot', val=np.ones(nn), desc='acceleration of equivalent airspeed', units='m/s**2')
        self.add_output(name='y', val=np.zeros(nn), desc='Lateral position along the trajectory', units='m')
        self.add_output(name='eas', val=np.ones(nn), desc='equivalent airspeed', units='m/s')
        self.add_output(name='n', val=np.ones(nn), desc='load factor', units=None)
        self.add_output(name='I_landing_gear', val=np.ones(nn), desc='flag for landing gear extraction', units=None)
        if phase_name == 'groundroll':
            self.add_output(name='v_rot_residual', val=np.ones(nn), desc='residual v - v_rot', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='v_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='D', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='L', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas', wrt='rho_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='D', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='L', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='rho_0', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='drho_0_dz', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='eas_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='L', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='gamma_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='x_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='x_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='z_dot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='z_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='L', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='n', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='I_landing_gear', wrt='z', rows=arange, cols=arange, val=1.0)
        if phase_name == 'groundroll':
            self.declare_partials(of='v_rot_residual', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='rho_0', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='c_l_max', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='k_rot', val=1.0)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        nn = self.options['num_nodes']
        settings = self.options['settings']
        phase_name = self.options['phase']
        ac = self.options['ac']
        atm = dict()
        atm['g'] = 9.80665
        atm['rho_0'] = 1.225

        # Load inputs
        v = inputs['v']
        F_n = ac.n_eng*inputs['F_n']
        L = inputs['L']
        D = inputs['D']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        rho_0 = inputs['rho_0']
        drho_0_dz = inputs['drho_0_dz']
        c_l_max = inputs['c_l_max']

        # Compute cosines and sines
        calpha = np.cos((ac.inc_F_n + alpha - ac.alpha_0)*np.pi/180.)
        salpha = np.sin((ac.inc_F_n + alpha - ac.alpha_0)*np.pi/180.)
        cgamma = np.cos(gamma*np.pi/180.)
        sgamma = np.sin(gamma*np.pi/180.)

        # Acceleration dvdt
        if phase_name in {'groundroll', 'rotation'}:
            F_fric = ac.mu_r * (ac.mtow * atm['g'] - L)
            outputs['v_dot'] = (1.0 / ac.mtow) * (F_n * calpha - D - F_fric) - atm['g'] * sgamma
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['v_dot'] = (1.0 / ac.mtow) * (F_n * calpha - D) - atm['g'] * sgamma

        # Change in position
        outputs['x_dot'] = v * cgamma
        outputs['z_dot'] = v * sgamma

        # Equivalent airspeed
        outputs['eas'] = v * np.sqrt(rho_0 / atm['rho_0'])
        outputs['eas_dot'] = outputs['v_dot'] * np.sqrt(rho_0 / atm['rho_0']) + v / 2 / np.sqrt(rho_0 * atm['rho_0']) * drho_0_dz * outputs['z_dot']

        # Climb angle
        if phase_name in {'groundroll', 'rotation'}:
            outputs['gamma_dot'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['gamma_dot'] = ((F_n * salpha + L - ac.mtow * atm['g'] * cgamma) / (ac.mtow * v))*180/np.pi

        # Compute net upward force
        outputs['n'] = (F_n * salpha + L)/(ac.mtow * atm['g'] * cgamma)

        # Stick (alpha) controller
        if phase_name in {'groundroll'}:
            outputs['alpha_dot'] = np.zeros(nn)
        elif phase_name == 'rotation':
            outputs['alpha_dot'] = 3.5 * np.ones(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['alpha_dot'] = np.zeros(nn)

        # Landing gear extraction
        if phase_name in {'groundroll', 'rotation', 'liftoff'}:
            outputs['I_landing_gear'] = np.ones(nn)
        elif phase_name in {'vnrs', 'cutback'}:
            outputs['I_landing_gear'] = np.zeros(nn)

        # Compute rotation speed residual
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * ac.mtow * atm['g'] / (ac.af_S_w * rho_0 * c_l_max))

            v_rot = inputs['k_rot'] * v_stall
            outputs['v_rot_residual'] = v - v_rot

        # Print to file
        if self.options['objective'] == 'noise':
            # Write k to file
            if phase_name == 'groundroll':
                f = open(settings.pyNA_directory + '/cases/' + settings.case_name + '/output/' + settings.output_directory_name + '/' + 'inputs_k.txt' , 'a')
                f.write(str(inputs['k_rot'][0]) + '\n')
                f.close()

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        ac = self.options['ac']

        atm = dict()
        atm['g'] = 9.80665
        atm['rho_0'] = 1.225

        # Load inputs
        v = inputs['v']
        F_n = ac.n_eng*inputs['F_n']
        L = inputs['L']
        D = inputs['D']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        rho_0 = inputs['rho_0']
        drho_0_dz = inputs['drho_0_dz']
        c_l_max = inputs['c_l_max']

        # Compute cosines and sines
        calpha = np.cos((ac.inc_F_n + alpha - ac.alpha_0)*np.pi/180.)
        salpha = np.sin((ac.inc_F_n + alpha - ac.alpha_0)*np.pi/180.)
        cgamma = np.cos(gamma*np.pi/180.)
        sgamma = np.sin(gamma*np.pi/180.)

        # Compute outputs
        F_fric = ac.mu_r * (ac.mtow * atm['g'] - L)
        zdot = v * sgamma

        # Acceleration dvdt
        if phase_name in {'groundroll', 'rotation'}:
            vdot = (1.0 / ac.mtow) * (F_n * calpha - D - F_fric - atm['g'] * ac.mtow * sgamma)
            partials['v_dot', 'F_n'] = calpha / ac.mtow * ac.n_eng
            partials['v_dot', 'D'] = - 1.0 / ac.mtow
            partials['v_dot', 'gamma'] = -atm['g'] * cgamma * np.pi/180.
            partials['v_dot', 'alpha'] = -F_n * salpha / ac.mtow * np.pi/180.
            partials['v_dot', 'L'] = ac.mu_r / ac.mtow
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            vdot = (1.0 / ac.mtow) * (F_n * calpha - D) - atm['g'] * sgamma
            partials['v_dot', 'F_n'] = calpha / ac.mtow * ac.n_eng
            partials['v_dot', 'D'] = - 1.0 / ac.mtow
            partials['v_dot', 'gamma'] = -atm['g'] * cgamma * np.pi/180.
            partials['v_dot', 'alpha'] = -F_n * salpha / ac.mtow * np.pi/180.
            partials['v_dot', 'L'] = 0.0

        # Equivalent airspeed
        partials['eas', 'v'] = np.sqrt(rho_0 / 1.225)
        partials['eas', 'rho_0'] = v / 2 / np.sqrt(rho_0 * 1.225)

        partials['eas_dot', 'F_n'] = partials['v_dot', 'F_n'] * np.sqrt(rho_0 / atm['rho_0'])
        partials['eas_dot', 'D'] = partials['v_dot', 'D'] * np.sqrt(rho_0 / atm['rho_0'])
        partials['eas_dot', 'gamma'] = partials['v_dot', 'gamma'] * np.sqrt(rho_0 / atm['rho_0']) + v ** 2 / 2. / np.sqrt(rho_0 * atm['rho_0']) * drho_0_dz * cgamma * np.pi/180.
        partials['eas_dot', 'alpha'] = partials['v_dot', 'alpha'] * np.sqrt(rho_0 / atm['rho_0'])
        partials['eas_dot', 'L'] = partials['v_dot', 'L'] * np.sqrt(rho_0 / atm['rho_0'])
        partials['eas_dot', 'rho_0'] = 1 / 2. / np.sqrt(rho_0 * atm['rho_0']) * vdot - v ** 2 / 4. / np.sqrt(rho_0 ** 3 * atm['rho_0']) * drho_0_dz * sgamma
        partials['eas_dot', 'drho_0_dz'] = v / 2 / np.sqrt(rho_0 * atm['rho_0']) * zdot
        partials['eas_dot', 'v'] = 1 / np.sqrt(rho_0 * atm['rho_0']) * drho_0_dz * zdot

        # Change in position
        partials['x_dot', 'gamma'] = -v * sgamma * np.pi / 180.
        partials['x_dot', 'v'] = cgamma

        partials['z_dot', 'gamma'] = v * cgamma * np.pi/180.
        partials['z_dot', 'v'] = sgamma

        # Change in climb angle
        if phase_name in {'groundroll', 'rotation'}:
            partials['gamma_dot', 'F_n'] = np.zeros(nn)
            partials['gamma_dot', 'L'] = np.zeros(nn)
            partials['gamma_dot', 'gamma'] = np.zeros(nn)
            partials['gamma_dot', 'alpha'] = np.zeros(nn)
            partials['gamma_dot', 'v'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            partials['gamma_dot', 'F_n'] = (salpha / (ac.mtow * v))*180/np.pi * ac.n_eng
            partials['gamma_dot', 'L'] = (1.0 / (ac.mtow * v))*180/np.pi
            partials['gamma_dot', 'gamma'] = (atm['g'] * sgamma * np.pi/180. / v)*180/np.pi
            partials['gamma_dot', 'alpha'] = (F_n * calpha * np.pi/180. / (ac.mtow * v))*180/np.pi
            partials['gamma_dot', 'v'] = (atm['g'] * cgamma / v ** 2 - (L + F_n * salpha) / (v ** 2 * ac.mtow))*180/np.pi

        # Net upward force
        partials['n', 'L'] = 1.0 / (ac.mtow * atm['g'] * cgamma)
        partials['n', 'F_n'] = salpha * ac.n_eng / (ac.mtow * atm['g'] * cgamma)
        partials['n', 'alpha'] = F_n * calpha * (np.pi/180.) / (ac.mtow * atm['g'] * cgamma)
        partials['n', 'gamma'] = (F_n * salpha + L) / (ac.mtow * atm['g']) / cgamma**2 * sgamma * (np.pi/180.)

        # Landing gear extraction
        partials['I_landing_gear', 'z'] = np.zeros(nn)

        # Compute rotation speed residual (v_residual = v - (inputs['k_rot'] * v_stall))
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * ac.mtow * atm['g'] / (ac.af_S_w * rho_0 * c_l_max))
            partials['v_rot_residual', 'v'] = np.ones(nn)
            partials['v_rot_residual', 'rho_0'] = inputs['k_rot'] / 2 / v_stall * 2 * ac.mtow * atm['g'] / (ac.af_S_w * rho_0 * c_l_max)**2 * (ac.af_S_w * c_l_max)
            partials['v_rot_residual', 'c_l_max'] = inputs['k_rot'] / 2 / v_stall * 2 * ac.mtow * atm['g'] / (ac.af_S_w * rho_0 * c_l_max)**2 * (ac.af_S_w * rho_0)
            partials['v_rot_residual', 'k_rot'] = - v_stall

import pdb
import openmdao
import datetime as dt
import numpy as np
from openmdao.api import ExplicitComponent
from pyNA.src.airframe import Airframe


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
        self.options.declare('airframe', types=Airframe)
        self.options.declare('sealevel_atmosphere', types=dict)
        self.options.declare('objective', types=str)
        self.options.declare('case_name', types=str)
        self.options.declare('output_directory_name', types=str)

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
        self.add_input(name='L', val=np.ones(nn), desc='lift', units='N')
        self.add_input(name='D', val=np.ones(nn), desc='drag', units='N')
        self.add_input(name='c_l_max', val=np.ones(nn), desc='maximum aircraft lift coefficient', units=None)
        if phase_name == 'groundroll':
            self.add_input(name='k_rot', val=1.2, desc='rotational speed parameter', units=None)

        # Outputs
        self.add_output(name='x_dot', val=np.ones(nn), desc='Position rate along the ground', units='m/s')
        self.add_output(name='z_dot', val=np.ones(nn), desc='rate of change of altitude', units='m/s')
        self.add_output(name='alpha_dot', val=np.ones(nn), desc='rate of change of angle of attack', units='deg/s')
        self.add_output(name='gamma_dot', val=np.ones(nn), desc='rate of change of flight path angle', units='deg/s')
        self.add_output(name='v_dot', val=np.ones(nn), desc='Acceleration along the ground (assuming zero wind)', units='m/s**2')        
        self.add_output(name='n', val=np.ones(nn), desc='load factor', units=None)
        if phase_name == 'groundroll':
            self.add_output(name='v_rot_residual', val=np.ones(nn), desc='residual v - v_rot', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='v_dot', wrt='F_n', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='D', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='gamma', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='alpha', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='v_dot', wrt='L', rows=arange, cols=arange, val=1.0)
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
        if phase_name == 'groundroll':
            self.declare_partials(of='v_rot_residual', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='c_l_max', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='v_rot_residual', wrt='k_rot', val=1.0)

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        airframe =  self.options['airframe']
        
        # Load inputs
        v = inputs['v']
        F_n = airframe.n_eng*inputs['F_n']
        L = inputs['L']
        D = inputs['D']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        c_l_max = inputs['c_l_max']

        # Compute cosines and sines
        calpha = np.cos((airframe.inc_F_n + alpha - airframe.alpha_0)*np.pi/180.)
        salpha = np.sin((airframe.inc_F_n + alpha - airframe.alpha_0)*np.pi/180.)
        cgamma = np.cos(gamma*np.pi/180.)
        sgamma = np.sin(gamma*np.pi/180.)

        # Acceleration dvdt
        if phase_name in {'groundroll', 'rotation'}:
            F_fric = airframe.mu_r * (airframe.mtow * self.options['sealevel_atmosphere']['g'] - L)
            outputs['v_dot'] = (1.0 / airframe.mtow) * (F_n * calpha - D - F_fric) - self.options['sealevel_atmosphere']['g'] * sgamma
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['v_dot'] = (1.0 / airframe.mtow) * (F_n * calpha - D) - self.options['sealevel_atmosphere']['g'] * sgamma

        # Change in position
        outputs['x_dot'] = v * cgamma
        outputs['z_dot'] = v * sgamma

        # Climb angle
        if phase_name in {'groundroll', 'rotation'}:
            outputs['gamma_dot'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['gamma_dot'] = ((F_n * salpha + L - airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma) / (airframe.mtow * v))*180/np.pi

        # Compute net upward force
        outputs['n'] = (F_n * salpha + L)/(airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma)

        # Stick (alpha) controller
        if phase_name in {'groundroll'}:
            outputs['alpha_dot'] = np.zeros(nn)
        elif phase_name in {'rotation'}:
            outputs['alpha_dot'] = 3.5 * np.ones(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            outputs['alpha_dot'] = np.zeros(nn)

        # Compute rotation speed residual
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * airframe.mtow * self.options['sealevel_atmosphere']['g'] / (airframe.af_S_w * self.options['sealevel_atmosphere']['rho_0'] * c_l_max))
            v_rot = inputs['k_rot'] * v_stall
            outputs['v_rot_residual'] = v - v_rot

        # Print to file
        if self.options['objective'] == 'noise':
            # Write k to file
            if phase_name == 'groundroll':
                f = open('/Users/laurensvoet/Documents/Research/pyNA/pyNA/cases/' + self.options['case_name'] + '/output/' + self.options['output_directory_name'] + '/' + 'inputs_k.txt' , 'a')
                f.write(str(inputs['k_rot'][0]) + '\n')
                f.close()

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        airframe =  self.options['airframe']

        # Load inputs
        v = inputs['v']
        F_n = airframe.n_eng*inputs['F_n']
        L = inputs['L']
        D = inputs['D']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        c_l_max = inputs['c_l_max']

        # Compute cosines and sines
        calpha = np.cos((airframe.inc_F_n + alpha - airframe.alpha_0)*np.pi/180.)
        c2alpha = np.cos(2*(airframe.inc_F_n + alpha - airframe.alpha_0)*np.pi/180.)
        salpha = np.sin((airframe.inc_F_n + alpha - airframe.alpha_0)*np.pi/180.)
        cgamma = np.cos(gamma*np.pi/180.)
        c2gamma = np.cos(2*gamma*np.pi/180.)
        sgamma = np.sin(gamma*np.pi/180.)

        # Compute outputs
        F_fric = airframe.mu_r * (airframe.mtow * self.options['sealevel_atmosphere']['g'] - L)
        zdot = v * sgamma

        # Acceleration dvdt
        if phase_name in {'groundroll', 'rotation'}:
            v_dot = (1.0 / airframe.mtow) * (F_n * calpha - D - F_fric) - self.options['sealevel_atmosphere']['g'] * sgamma
            partials['v_dot', 'F_n'] = calpha / airframe.mtow * airframe.n_eng
            partials['v_dot', 'D'] = - 1.0 / airframe.mtow
            partials['v_dot', 'gamma'] = -self.options['sealevel_atmosphere']['g'] * cgamma * np.pi/180.
            partials['v_dot', 'alpha'] = -F_n * salpha / airframe.mtow * np.pi/180.
            partials['v_dot', 'L'] = airframe.mu_r / airframe.mtow
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            v_dot = (1.0 / airframe.mtow) * (F_n * calpha - D) - self.options['sealevel_atmosphere']['g'] * sgamma
            partials['v_dot', 'F_n'] = calpha / airframe.mtow * airframe.n_eng
            partials['v_dot', 'D'] = - 1.0 / airframe.mtow
            partials['v_dot', 'gamma'] = -self.options['sealevel_atmosphere']['g'] * cgamma * np.pi/180.
            partials['v_dot', 'alpha'] = -F_n * salpha / airframe.mtow * np.pi/180.
            partials['v_dot', 'L'] = 0.0

        # Change in position
        partials['x_dot', 'gamma'] = -v * sgamma * np.pi / 180.
        partials['x_dot', 'v'] = cgamma

        partials['z_dot', 'gamma'] = v * cgamma * np.pi/180.
        partials['z_dot', 'v'] = sgamma

        # Change in climb angle
        if phase_name in {'groundroll', 'rotation'}:
            gamma_dot = np.zeros(nn)
            partials['gamma_dot', 'F_n'] = np.zeros(nn)
            partials['gamma_dot', 'L'] = np.zeros(nn)
            partials['gamma_dot', 'gamma'] = np.zeros(nn)
            partials['gamma_dot', 'alpha'] = np.zeros(nn)
            partials['gamma_dot', 'v'] = np.zeros(nn)
        elif phase_name in {'liftoff', 'vnrs', 'cutback'}:
            gamma_dot = ((F_n * salpha + L - airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma) / (airframe.mtow * v))*180/np.pi
            partials['gamma_dot', 'F_n'] = (salpha / (airframe.mtow * v))*180/np.pi * airframe.n_eng
            partials['gamma_dot', 'L'] = (1.0 / (airframe.mtow * v))*180/np.pi
            partials['gamma_dot', 'gamma'] = (self.options['sealevel_atmosphere']['g'] * sgamma * np.pi/180. / v)*180/np.pi
            partials['gamma_dot', 'alpha'] = (F_n * calpha * np.pi/180. / (airframe.mtow * v))*180/np.pi
            partials['gamma_dot', 'v'] = (self.options['sealevel_atmosphere']['g'] * cgamma / v ** 2 - (L + F_n * salpha) / (v ** 2 * airframe.mtow))*180/np.pi

        # Net upward force
        partials['n', 'L'] = 1.0 / (airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma)
        partials['n', 'F_n'] = salpha * airframe.n_eng / (airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma)
        partials['n', 'alpha'] = F_n * calpha * (np.pi/180.) / (airframe.mtow * self.options['sealevel_atmosphere']['g'] * cgamma)
        partials['n', 'gamma'] = (F_n * salpha + L) / (airframe.mtow * self.options['sealevel_atmosphere']['g']) / cgamma**2 * sgamma * (np.pi/180.)

        # Compute rotation speed residual (v_residual = v - (airframe.k_rot * v_stall))
        if phase_name == 'groundroll':
            v_stall = np.sqrt(2 * airframe.mtow * self.options['sealevel_atmosphere']['g'] / (airframe.af_S_w * self.options['sealevel_atmosphere']['rho_0'] * c_l_max))
            partials['v_rot_residual', 'v'] = np.ones(nn)
            partials['v_rot_residual', 'c_l_max'] = inputs['k_rot'] / 2 / v_stall * 2 * airframe.mtow * self.options['sealevel_atmosphere']['g'] / (airframe.af_S_w * self.options['sealevel_atmosphere']['rho_0'] * c_l_max)**2 * (airframe.af_S_w * self.options['sealevel_atmosphere']['rho_0'])
            partials['v_rot_residual', 'k_rot'] = - v_stall

import pdb
import numpy as np
import openmdao
import openmdao.api as om


class TrajectoryData(om.ExplicitComponent):
    """
    Create trajectory model from time history data.

    The *TrajectoryData* component computes the following outputs:

    * ``outputs['x']``:             distance past brake release point [m]
    * ``outputs['y']``:             lateral distance from center line [m]
    * ``outputs['z']``:             altitude [m]
    * ``outputs['alpha']``:         aircraft angle of attack [deg]
    * ``outputs['gamma']``:         aircraft climb angle [deg]
    * ``outputs['t_s']``:           source time [s]
    * ``outputs['tau']``:           thrust setting [-]
    * ``outputs['M_0']``:           aircraft flight Mach number [-]
    * ``outputs['c_0']``:           ambient speed of sound [m/s]
    * ``outputs['T_0']``:           ambient temperature [K]
    * ``outputs['p_0']``:           ambient pressure [Pa]
    * ``outputs['rho_0']``:         ambient density [kg/m3]
    * ``outputs['mu_0']``:          ambient dynamic viscosity [kg/ms]
    * ``outputs['I_0']``:           ambient characteristic impedance [kg/m2s]
    * ``outputs['theta_flaps']``:   aircraft flap deflection angle [deg]
    * ``outputs['I_lg']``:          aircraft landing gear retraction (0)/ extension (1)

    * ``outputs['jet_V']``:         engine jet velocity [m/s]
    * ``outputs['jet_rho']``:       engine jet density [kg/m3]
    * ``outputs['jet_A']``:         engine jet area [m2]
    * ``outputs['jet_Tt']``:        engine jet total temperature [K]
    * ``outputs['jet_M']``:         engine jet Mach number

    * ``outputs['core_mdot']``:     engine core inlet mass flow [kg/s] 
    * ``outputs['core_Tt_i']``:     engine core inlet total temperature [K]
    * ``outputs['core_Tt_j']``:     engine core exit total temperature [K]
    * ``outputs['core_Pt_i']``:     engine core inlet pressure [Pa]
    * ``outputs['turb_DTt_des']``:  engine turbine temperature drop at design [K]
    * ``outputs['turb_rho_i']``:    engine turbine inlet density [kg/m3]
    * ``outputs['turb_c_i']``:      engine turbine inlet speed of sound [m/s]
    * ``outputs['turb_rho_e']``:    engine turbine exit density [kg/m3]
    * ``outputs['turb_c_e']``:      engine turbine exit speed of sound [m/s]

    * ``outputs['fan_DTt']``:       engine fan temperature rise [K]
    * ``outputs['fan_mdot']``:      engine fan inlet mass flow [kg/m3]
    * ``outputs['fan_N']``:         engine fan rotational speed [rpm]

    """

    def initialize(self):
        # Declare data option
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('settings', types=dict, desc='pyNA settings')

    def setup(self):

        # Load options
        nn = self.options['num_nodes']
        settings=self.options['settings']

        # Trajectory variables
        self.add_output('t_s', val=np.ones(nn), units='s',desc='source time')
        self.add_output('x', val=np.ones(nn), units='m',desc='distance past brake release point')
        self.add_output('y', val=np.ones(nn), units='m',desc='lateral distance from center line')
        self.add_output('z', val=np.ones(nn), units='m',desc='altitude')
        self.add_output('v', val=np.ones(nn), units='m/s',desc='aircraft velocity')
        self.add_output('alpha', val=np.ones(nn), units='deg',desc='aircraft angle of attack')
        self.add_output('gamma', val=np.ones(nn), units='deg',desc='aircraft climb angle')
        self.add_output('F_n', val=np.ones(nn), units=None,desc='aircraft net thrust')
        self.add_output('tau', val=np.ones(nn), units=None,desc='thrust setting')
        self.add_output('M_0', val=np.ones(nn), units=None,desc='aircraft flight Mach number')
        self.add_output('c_0', val=np.ones(nn), units='m/s',desc='ambient speed of sound')
        self.add_output('T_0', val=np.ones(nn), units='K',desc='ambient temperature')
        self.add_output('p_0', val=np.ones(nn), units='Pa',desc='ambient pressure')
        self.add_output('rho_0', val=np.ones(nn), units='kg/m**3',desc='ambient density')
        self.add_output('mu_0', val=np.ones(nn), units='kg/m/s',desc='ambient dynamic viscosity')
        self.add_output('I_0', val=np.ones(nn), units='kg/m**2/s',desc='ambient characteristic impedance')

        if settings['airframe_source']:
            self.add_output('theta_flaps', val=np.ones(nn), units='deg',desc='aircraft flap deflection angle')
            self.add_output('I_lg', val=np.ones(nn), units=None,desc='lircraft landing gear retraction (0)/ extension (1)')

        # Engine variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            self.add_output('jet_V', val=np.ones(nn), units='m/s',desc='engine jet velocity')
            self.add_output('jet_rho', val=np.ones(nn), units='kg/m**3',desc='engine jet density')
            self.add_output('jet_A', val=np.ones(nn), units='m**2',desc='engine jet area')
            self.add_output('jet_Tt', val=np.ones(nn), units='K',desc='engine jet total temperature')
        elif not settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.add_output('jet_V', val=np.ones(nn), units='m/s',desc='engine jet velocity')
            self.add_output('jet_A', val=np.ones(nn), units='m**2',desc='engine jet area')
            self.add_output('jet_Tt', val=np.ones(nn), units='K',desc='engine jet total temperature')
            self.add_output('jet_M', val=np.ones(nn), units=None,desc='engine jet Mach number')
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.add_output('jet_V', val=np.ones(nn), units='m/s',desc='engine jet velocity')
            self.add_output('jet_rho', val=np.ones(nn), units='kg/m**3',desc='engine jet density')
            self.add_output('jet_A', val=np.ones(nn), units='m**2',desc='engine jet area')
            self.add_output('jet_Tt', val=np.ones(nn), units='K',desc='engine jet total temperature')
            self.add_output('jet_M', val=np.ones(nn), units=None,desc='engine jet Mach number')
        
        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == "ge":
                self.add_output('core_mdot', val=np.ones(nn), units='kg/s', desc='engine core massflow')
                self.add_output('core_Tt_i', val=np.ones(nn), units='K', desc='engine core inlet total temperature')
                self.add_output('core_Tt_j', val=np.ones(nn), units='K', desc='engine core exit total temperature')
                self.add_output('core_Pt_i', val=np.ones(nn), units='Pa', desc='engine core inlet total pressure')
                self.add_output('turb_DTt_des', val=np.ones(nn), units='K', desc='engine turbine temperature drop at design')
            if settings['core_turbine_attenuation_method'] == "pw":
                self.add_output('core_mdot', val=np.ones(nn), units='kg/s', desc='engine core massflow')
                self.add_output('core_Tt_i', val=np.ones(nn), units='K', desc='engine core inlet total temperature')
                self.add_output('core_Tt_j', val=np.ones(nn), units='K', desc='engine core exit total temperature')
                self.add_output('core_Pt_i', val=np.ones(nn), units='Pa', desc='engine core inlet total pressure')        
                self.add_output('turb_rho_i', val=np.ones(nn), units='kg/m**3', desc='engine turbine inlet density')
                self.add_output('turb_c_i', val=np.ones(nn), units='m/s', desc='engine turbine inlet speed of sound')
                self.add_output('turb_rho_e', val=np.ones(nn), units='kg/m**3', desc='engine turbine exit density')
                self.add_output('turb_c_e', val=np.ones(nn), units='m/s', desc='engine turbine exit speed of sound')

        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            self.add_output('fan_DTt', val=np.ones(nn), units='K', desc='engine fan total temperature rise')
            self.add_output('fan_mdot', val=np.ones(nn), units='kg/s', desc='engine fan inlet mass flow')
            self.add_output('fan_N', val=np.ones(nn), units='rpm', desc='engine fan rotational speed')
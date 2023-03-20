import openmdao.api as om
import numpy as np
from pyNA.src.noise_model.tables import Tables
from pyNA.src.aircraft import Aircraft
from pyNA.src.noise_model.python.calculate_noise import calculate_noise

class NoiseModel(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('settings', types=dict, desc='Pyna settings')
        self.options.declare('aircraft', types=Aircraft, desc='')
        self.options.declare('tables', types=Tables, desc='')
        self.options.declare('f', types=np.ndarray, desc='1/3rd octave frequency bands')
        self.options.declare('f_sb', types=np.ndarray, desc='1/3rd octave frequency sub-bands')
        self.options.declare('n_t', types=int, desc='Number of time steps')
        self.options.declare('optimization', types=bool, desc='Flag')

    def setup(self):

        n_t = self.options['n_t'] 
        settings = self.options['settings']

        self.add_input(name='x', val=np.ones(n_t, ), desc='position along the trajectory', units='m')
        self.add_input(name='y', val=np.ones(n_t, ), desc='lateral position from the centerline of the trajectory', units='m')
        self.add_input(name='z', val=np.ones(n_t, ), desc='altitude', units='m')
        self.add_input(name='alpha', val=np.ones(n_t, ), desc='angle of attack', units='deg')
        self.add_input(name='gamma', val=np.ones(n_t, ), desc='flight path angle', units='deg')
        self.add_input(name='t_s', val=np.ones(n_t, ), desc='source time', units='s')
        self.add_input(name='tau', val=np.ones(n_t, ), desc='thrust-setting', units=None)
        self.add_input(name='M_0', val=np.ones(n_t, ), desc='flight Mach number', units=None)
        self.add_input(name='c_0', val=np.ones(n_t, ), desc='ambient speed of sound', units='m/s')
        self.add_input(name='T_0', val=np.ones(n_t, ), desc='ambient temperature', units='K')
        self.add_input(name='rho_0', val=np.ones(n_t, ), desc='ambient density', units='kg/m**3')
        self.add_input(name='P_0', val=np.ones(n_t, ), desc='ambient pressure', units='Pa')
        self.add_input(name='mu_0', val=np.ones(n_t, ), desc='ambient dynamic viscosity', units='kg/m/s')
        self.add_input(name='I_0', val=np.ones(n_t, ), units='kg/m**2/s', desc='ambient characteristic impedance')
        
        self.add_input(name='fan_DTt', val=np.ones(n_t, ), units='K', desc='fan total temperature rise')
        self.add_input(name='fan_mdot', val=np.ones(n_t, ), units='kg/s', desc='fan mass flow rate')
        self.add_input(name='fan_N', val=np.ones(n_t, ), units='rpm', desc='fan rotational speed')
        self.add_input(name='core_mdot', val=np.ones(n_t, ), units='kg/s', desc='core mass flow rate')
        self.add_input(name='core_Tt_i', val=np.ones(n_t, ), units='K', desc='combuster inlet total temperature')
        self.add_input(name='core_Tt_j', val=np.ones(n_t, ), units='K', desc='combuster inlet total temperature')
        self.add_input(name='core_Pt_i', val=np.ones(n_t, ), units='Pa', desc='combuster inlet total pressure')
        self.add_input(name='turb_DTt_des', val=np.ones(n_t, ), units='K', desc='turbine total temperature drop at the design point')
        self.add_input(name='turb_rho_e', val=np.ones(n_t, ), units='kg/m**3', desc='turbine exit density')
        self.add_input(name='turb_c_e', val=np.ones(n_t, ), units='m/s', desc='turbine exit speed of sound')
        self.add_input(name='turb_rho_i', val=np.ones(n_t, ), units='kg/m**3', desc='turbine inlet density')
        self.add_input(name='turb_c_i', val=np.ones(n_t, ), units='m/s', desc='turbine inlet speed of sound')
        self.add_input(name='jet_V', val=np.ones(n_t, ), units='m/s', desc='jet velocity')
        self.add_input(name='jet_rho', val=np.ones(n_t, ), units='kg/m**3', desc='jet density')
        self.add_input(name='jet_A', val=np.ones(n_t, ), units='m**2', desc='jet area')
        self.add_input(name='jet_Tt', val=np.ones(n_t, ), units='K', desc='jet total temperature')
        self.add_input(name='jet_M', val=np.ones(n_t, ), units=None, desc='jet Mach number')
        self.add_input(name='theta_flaps', val=np.ones(n_t, ), units='deg', desc='flap deflection angle')
        self.add_input(name='I_lg', val=np.ones(n_t, ), units=None, desc='flag for landing gear extraction (1: yes; 0: no)')

        n_obs = settings["x_microphones"].shape[0]
        if self.options['optimization']:
            self.add_output(name='level_int', val=np.ones(n_obs,))
        else:
            self.add_output(name='t_o', val=np.ones((n_obs, n_t,)))
            self.add_output(name='spl', val=np.ones((n_obs, n_t, settings['n_frequency_bands'],)))
            self.add_output(name='level', val=np.ones((n_obs, n_t,)))
            self.add_output(name='level_int', val=np.ones((n_obs,)))

    def setup_partials(self):

        if self.options['optimization']:
                        
            self.declare_partials(of='level_int', wrt='x')
            self.declare_partials(of='level_int', wrt='y')
            self.declare_partials(of='level_int', wrt='z')
            self.declare_partials(of='level_int', wrt='alpha')
            self.declare_partials(of='level_int', wrt='gamma')
            self.declare_partials(of='level_int', wrt='t_s')
            self.declare_partials(of='level_int', wrt='tau')
            self.declare_partials(of='level_int', wrt='M_0')
            self.declare_partials(of='level_int', wrt='c_0')
            self.declare_partials(of='level_int', wrt='T_0')
            self.declare_partials(of='level_int', wrt='rho_0')
            self.declare_partials(of='level_int', wrt='P_0')
            self.declare_partials(of='level_int', wrt='mu_0')
            self.declare_partials(of='level_int', wrt='I_0')
            self.declare_partials(of='level_int', wrt='fan_DTt')
            self.declare_partials(of='level_int', wrt='fan_mdot')
            self.declare_partials(of='level_int', wrt='fan_N')
            self.declare_partials(of='level_int', wrt='core_mdot')
            self.declare_partials(of='level_int', wrt='core_Tt_i')
            self.declare_partials(of='level_int', wrt='core_Tt_j')
            self.declare_partials(of='level_int', wrt='core_Pt_i')
            self.declare_partials(of='level_int', wrt='turb_DTt_des')
            self.declare_partials(of='level_int', wrt='turb_rho_e')
            self.declare_partials(of='level_int', wrt='turb_c_e')
            self.declare_partials(of='level_int', wrt='turb_rho_i')
            self.declare_partials(of='level_int', wrt='turb_c_i')
            self.declare_partials(of='level_int', wrt='jet_V')
            self.declare_partials(of='level_int', wrt='jet_rho')
            self.declare_partials(of='level_int', wrt='jet_A')
            self.declare_partials(of='level_int', wrt='jet_Tt')
            self.declare_partials(of='level_int', wrt='jet_M')
            self.declare_partials(of='level_int', wrt='theta_flaps')
            self.declare_partials(of='level_int', wrt='I_lg')

    def compute(self, inputs, outputs):
        
        settings = self.options['settings']
        f = self.options['f']
        f_sb = self.options['f_sb']
        aircraft = self.options['aircraft']
        tables = self.options['tables']

        if self.options['optimization']:
            outputs['level_int'] = calculate_noise(inputs['x'], inputs['y'], inputs['z'], inputs['alpha'], inputs['gamma'], inputs['t_s'], inputs['tau'], 
                                                   inputs['M_0'], inputs['c_0'], inputs['T_0'], inputs['rho_0'], inputs['P_0'], inputs['mu_0'], inputs['I_0'],
                                                   inputs['fan_DTt'], inputs['fan_mdot'], inputs['fan_N'], 
                                                   inputs['core_mdot'], inputs['core_Tt_i'], inputs['core_Tt_j'], inputs['core_Pt_i'], inputs['turb_DTt_des'], inputs['turb_rho_e'], inputs['turb_c_e'], inputs['turb_rho_i'], inputs['turb_c_i'],
                                                   inputs['jet_V'], inputs['jet_rho'], inputs['jet_A'], inputs['jet_Tt'], inputs['jet_M'], 
                                                   inputs['theta_flaps'], inputs['I_lg'],  
                                                   x_microphone,
                                                   f, f_sb, 
                                                   settings, aircraft, tables)
            
        else:
            outputs['t_o'], outputs['spl'], outputs['level'], outputs['level_int'] = calculate_noise(inputs['x'], inputs['y'], inputs['z'], inputs['alpha'], inputs['gamma'], inputs['t_s'], inputs['tau'], 
                                                                                                     inputs['M_0'], inputs['c_0'], inputs['T_0'], inputs['rho_0'], inputs['P_0'], inputs['mu_0'], inputs['I_0'],
                                                                                                     inputs['fan_DTt'], inputs['fan_mdot'], inputs['fan_N'], 
                                                                                                     inputs['core_mdot'], inputs['core_Tt_i'], inputs['core_Tt_j'], inputs['core_Pt_i'], inputs['turb_DTt_des'], inputs['turb_rho_e'], inputs['turb_c_e'], inputs['turb_rho_i'], inputs['turb_c_i'],
                                                                                                     inputs['jet_V'], inputs['jet_rho'], inputs['jet_A'], inputs['jet_Tt'], inputs['jet_M'], 
                                                                                                     inputs['theta_flaps'], inputs['I_lg'], 
                                                                                                     x_microphone,
                                                                                                     f, f_sb, 
                                                                                                     settings, aircraft, tables)

        return

    def compute_partials(self, inputs, partials):
        
        if self.options['optimization']:
            pass

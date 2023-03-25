import openmdao.api as om
import numpy as np
from pyNA.src.noise_model.tables import Tables
from pyNA.src.aircraft import Aircraft
from pyNA.src.noise_model.python.calculate_noise import calculate_noise
from pyNA.src.noise_model.python.source.calculate_shielding_factor import calculate_shielding_factor
import pdb


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
        n_obs = settings["x_microphones"].shape[0]

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
        self.add_input(name='I_landing_gear', val=np.ones(n_t, ), units=None, desc='flag for landing gear extraction (1: yes; 0: no)')

        if self.options['optimization']:
            self.add_output(name='level', val=np.ones(n_obs, n_t))
        else:
            self.add_output(name='t_o', val=np.ones((n_obs, n_t,)))
            self.add_output(name='spl', val=np.ones((n_obs, n_t, settings['n_frequency_bands'],)))
            self.add_output(name='aspl', val=np.ones((n_obs, n_t,)))
            self.add_output(name='oaspl', val=np.ones((n_obs, n_t,)))
            self.add_output(name='pnlt', val=np.ones((n_obs, n_t,)))
            # self.add_output(name='level_int', val=np.ones((n_obs,)))

    def setup_partials(self):

        pass

    def compute(self, inputs, outputs):
        
        settings = self.options['settings']
        f = self.options['f']
        f_sb = self.options['f_sb']
        aircraft = self.options['aircraft']
        tables = self.options['tables']
        n_t = self.options['n_t']
        n_obs = settings["x_microphones"].shape[0]

        if self.options['optimization']:
            
            for i in np.arange(n_obs):
                for j in np.arange(n_t):
                    x_microphone = settings['x_microphones'][i,:]

                    shielding = calculate_shielding_factor(settings)

                    outputs['t_o'][i, j], outputs['spl'][i, j, :], outputs['level'][i, j] = calculate_noise(inputs['x'][j], inputs['y'][j], inputs['z'][j], inputs['alpha'][j], inputs['gamma'][j], inputs['t_s'][j], inputs['tau'][j], 
                                                    inputs['M_0'][j], inputs['c_0'][j], inputs['T_0'][j], inputs['rho_0'][j], inputs['P_0'][j], inputs['mu_0'][j], inputs['I_0'][j],
                                                    inputs['fan_DTt'][j], inputs['fan_mdot'][j], inputs['fan_N'][j], 
                                                    inputs['core_mdot'][j], inputs['core_Tt_i'][j], inputs['core_Tt_j'][j], inputs['core_Pt_i'][j], inputs['turb_DTt_des'][j], inputs['turb_rho_e'][j], inputs['turb_c_e'][j], inputs['turb_rho_i'][j], inputs['turb_c_i'][j],
                                                    inputs['jet_V'][j], inputs['jet_rho'][j], inputs['jet_A'][j], inputs['jet_Tt'][j], inputs['jet_M'][j], 
                                                    inputs['theta_flaps'][j], inputs['I_landing_gear'][j],  
                                                    shielding,
                                                    x_microphone,
                                                    f, f_sb,
                                                    settings, aircraft, tables,
                                                    self.options['optimization'])

        else:

            for i in np.arange(n_obs):
                
                for j in np.arange(n_t):

                    x_microphone = settings['x_microphones'][i,:]
    
                    shielding = calculate_shielding_factor(settings, tables, i, j)

                    outputs['t_o'][i, j], outputs['spl'][i, j, :], outputs['aspl'][i, j], outputs['oaspl'][i, j], outputs['pnlt'][i, j] = calculate_noise(inputs['x'][j], inputs['y'][j], inputs['z'][j], inputs['alpha'][j], inputs['gamma'][j], inputs['t_s'][j], inputs['tau'][j], 
                                                    inputs['M_0'][j], inputs['c_0'][j], inputs['T_0'][j], inputs['rho_0'][j], inputs['P_0'][j], inputs['mu_0'][j], inputs['I_0'][j],
                                                    inputs['fan_DTt'][j], inputs['fan_mdot'][j], inputs['fan_N'][j], 
                                                    inputs['core_mdot'][j], inputs['core_Tt_i'][j], inputs['core_Tt_j'][j], inputs['core_Pt_i'][j], inputs['turb_DTt_des'][j], inputs['turb_rho_e'][j], inputs['turb_c_e'][j], inputs['turb_rho_i'][j], inputs['turb_c_i'][j],
                                                    inputs['jet_V'][j], inputs['jet_rho'][j], inputs['jet_A'][j], inputs['jet_Tt'][j], inputs['jet_M'][j], 
                                                    inputs['theta_flaps'][j], inputs['I_landing_gear'][j],  
                                                    shielding,
                                                    x_microphone,
                                                    f, f_sb,
                                                    settings, aircraft, tables,
                                                    self.options['optimization'])

        return

    def compute_partials(self, inputs, partials):
        
        pass

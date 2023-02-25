import pandas as pd
import numpy as np
import pyNA


class Engine:
    
    def __init__(self) -> None:
        
        self.deck = dict()

        self.var = list()
        self.var_units = dict()

    def get_var(self, settings) -> None:
        """
        Get list of required engine variables for noise calculations

        :return: None
        """

        # General variables
        self.var.append('F_n'); self.var_units['F_n'] = 'N'
        self.var.append('W_f'); self.var_units['W_f'] = 'kg/s'
        self.var.append('core_Tt_i'); self.var_units['core_Tt_i'] = 'K'
        self.var.append('core_Pt_i'); self.var_units['core_Pt_i'] = 'Pa'

        # Jet variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            self.var.append('jet_V'); self.var_units['jet_V'] = 'm/s' 
            self.var.append('jet_rho'); self.var_units['jet_rho'] = 'kg/m**3'
            self.var.append('jet_A'); self.var_units['jet_A'] = 'm**2'
            self.var.append('jet_Tt'); self.var_units['jet_Tt'] = 'K'
        elif settings['jet_shock_source'] and not settings['jet_mixing_source']:
            self.var.append('jet_V'); self.var_units['jet_V'] = 'm/s' 
            self.var.append('jet_A'); self.var_units['jet_A'] = 'm**2'
            self.var.append('jet_Tt'); self.var_units['jet_Tt'] = 'K'
            self.var.append('jet_M'); self.var_units['jet_M'] = None
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.var.append('jet_V'); self.var_units['jet_V'] = 'm/s' 
            self.var.append('jet_rho'); self.var_units['jet_rho'] = 'kg/m**3'
            self.var.append('jet_A'); self.var_units['jet_A'] = 'm**2'
            self.var.append('jet_Tt'); self.var_units['jet_Tt'] = 'K'
            self.var.append('jet_M'); self.var_units['jet_M'] = None
        
        # Core variables
        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == 'ge':
                self.var.append('core_mdot'); self.var_units['core_mdot'] = 'kg/s'
                self.var.append('core_Tt_j'); self.var_units['core_Tt_j'] = 'K'
                self.var.append('turb_DTt_des'); self.var_units['turb_DTt_des'] = 'K'

            elif settings['core_turbine_attenuation_method'].method_core_turb == 'pw':
                self.var.append('core_mdot'); self.var_units['core_mdot'] = 'kg/s'
                self.var.append('core_Tt_j'); self.var_units['core_Tt_j'] = 'K'
                self.var.append('turb_rho_e'); self.var_units['turb_rho_e'] = 'kg/m**3'
                self.var.append('turb_c_e'); self.var_units['turb_c_e', ] = 'm/s'
                self.var.append('turb_rho_i'); self.var_units['turb_rho_i'] = 'kg/m**3'
                self.var.append('turb_c_i'); self.var_units['turb_c_i'] = 'm/s'

        # Fan variables
        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            self.var.append('fan_DTt'); self.var_units['fan_DTt'] = 'K'
            self.var.append('fan_mdot'); self.var_units['fan_mdot'] = 'kg/s'
            self.var.append('fan_N'); self.var_units['fan_N'] = 'rpm'

        return None

    def get_performance_deck(self, settings) -> None:
        """
        Load engine deck for trajectory computations.

        :return: None
        """

        # Load self.engine data and create interpolation functions
        data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/aircraft/' + settings['engine_deck_file_name'])
        deck_labels = {'z':'z [m]',
                       'M_0':'M_0 [-]',
                       'tau':'tau [-]',
                       'F_n':'F_n [N]',
                       'W_f':'W_f [kg/s]',
                       'jet_V':'jet_V [m/s]',
                       'jet_Tt':'jet_Tt [K]',
                       'jet_rho':'jet_rho [kg/m3]',
                       'jet_A':'jet_A [m2]',
                       'jet_M':'jet_M [-]',
                       'core_mdot':'core_mdot [kg/s]',
                       'core_Pt_i':'core_Pt_i [Pa]',
                       'core_Tt_i':'core_Tt_i [K]',
                       'core_Tt_j':'core_Tt_j [K]',
                       'turb_DTt_des':'turb_DTt_des [K]',
                       'turb_rho_te':'turb_rho_e [kg/m3]',
                       'turb_rho_ti':'turb_rho_i [kg/m3]',
                       'turb_c_te':'turb_c_e [m/s]',
                       'turb_c_ti':'turb_c_i [m/s]',
                       'fan_DTt':'fan_DTt [K]',
                       'fan_mdot':'fan_mdot [kg/s]',
                       'fan_N':'fan_N [rpm]'}

        # Get engine variables and units
        Engine.get_var(self, settings)

        # Initialize engine deck variables
        if settings['atmosphere_mode'] == 'stratified':
            self.deck['z'] = np.unique(data[deck_labels['z']].values)
            self.deck['M_0'] = np.unique(data[deck_labels['M_0']].values)
            self.deck['TS'] = np.unique(data[deck_labels['tau']].values)
            
            if not len(self.var) == 0:
                for var in self.var:
                    self.deck[var] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

                # Fill engine deck
                cntr = -1
                for i in np.arange(self.deck['z'].shape[0]):
                    for j in np.arange(self.deck['M_0'].shape[0]):
                        for k in np.flip(np.arange(self.deck['TS'].shape[0])):
                            cntr = cntr + 1

                            for var in self.var:

                                if var == 'F_n':
                                    if settings['thrust_lapse']:
                                        self.deck[var][i, j, k] = data[deck_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = settings['F00']

                                else:
                                    if settings['thrust_lapse']:
                                        self.deck[var][i, j, k] = data[deck_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = data[deck_labels[var]].values[1]
            else:
                raise ValueError('deck_var list is empty') 

        elif settings['atmosphere_mode'] == 'sealevel':

            # Select only sealevel values in engine deck
            data = data[data['z [m]'] == 0]

            self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
            self.deck['TS'] = np.unique(data['tau [-]'].values)

            if not len(self.var) == 0:
                for var in self.var:
                    self.deck[var] = np.zeros((self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

                # Fill engine deck
                cntr = -1
                for j in np.arange(self.deck['M_0'].shape[0]):
                    for k in np.flip(np.arange(self.deck['TS'].shape[0])):
                        cntr = cntr + 1

                        for var in self.var:

                            if var == 'F_n':
                                if settings['thrust_lapse']:
                                    self.deck[var][j, k] = data[deck_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = settings['F00']

                            else:
                                if settings['thrust_lapse']:
                                    self.deck[var][j, k] = data[deck_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = data[deck_labels[var]].values[1]

            else:
                raise ValueError('deck_var list is empty') 

        return None
 
    # def load_timeseries_operating_point(self, timestep):

    #     # Select operating point
    #     cols = self.timeseries.columns
    #     op_point = pd.DataFrame(np.reshape(self.timeseries.values[timestep, :], (1, len(cols))))
    #     op_point.columns = cols

    #     # Duplicate operating for theta range (np.linspace(0, 180, 19))
    #     self.timeseries = pd.DataFrame()
    #     for _ in np.arange(19):
    #         self.timeseries = self.timeseries.append(op_point)

    #     return None
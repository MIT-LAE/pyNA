import pandas as pd
import numpy as np


class Engine:
    
    def __init__(self) -> None:
        
        self.var = list()
        self.var_units = dict()
        self.deck = dict()
        self.timeseries = pd.DataFrame()

    def get_source_noise_variables(self, settings) -> None:
        """
        Get list of required engine variables for noise calculations

        :return: None
        """

        # General variables
        self.var.append('F_n'); self.var_units['F_n'] = 'N'
        self.var.append('W_f'); self.var_units['W_f'] = 'kg/s'
        self.var.append('Tti_c'); self.var_units['Tti_c'] = 'K'
        self.var.append('Pti_c'); self.var_units['Pti_c'] = 'Pa'

        # Jet variables
        if settings['jet_mixing_source'] and not settings['jet_shock_source']:
            self.var.append('V_j'); self.var_units['V_j'] = 'm/s' 
            self.var.append('rho_j'); self.var_units['rho_j'] = 'kg/m**3'
            self.var.append('A_j'); self.var_units['A_j'] = 'm**2'
            self.var.append('Tt_j'); self.var_units['Tt_j'] = 'K'
        elif settings['jet_shock_source'] and not settings['jet_mixing_source']:
            self.var.append('V_j'); self.var_units['V_j'] = 'm/s' 
            self.var.append('A_j'); self.var_units['A_j'] = 'm**2'
            self.var.append('Tt_j'); self.var_units['Tt_j'] = 'K'
            self.var.append('M_j'); self.var_units['M_j'] = None
        elif settings['jet_mixing_source'] and settings['jet_shock_source']:
            self.var.append('V_j'); self.var_units['V_j'] = 'm/s' 
            self.var.append('rho_j'); self.var_units['rho_j'] = 'kg/m**3'
            self.var.append('A_j'); self.var_units['A_j'] = 'm**2'
            self.var.append('Tt_j'); self.var_units['Tt_j'] = 'K'
            self.var.append('M_j'); self.var_units['M_j'] = None
        
        # Core variables
        if settings['core_source']:
            if settings['core_turbine_attenuation_method'] == 'ge':
                self.var.append('mdoti_c'); self.var_units['mdoti_c'] = 'kg/s'
                self.var.append('Ttj_c'); self.var_units['Ttj_c'] = 'K'
                self.var.append('DTt_des_c'); self.var_units['DTt_des_c'] = 'K'

            elif settings['core_turbine_attenuation_method'].method_core_turb == 'pw':
                self.var.append('mdoti_c'); self.var_units['mdoti_c'] = 'kg/s'
                self.var.append('Ttj_c'); self.var_units['Ttj_c'] = 'K'
                self.var.append('DTt_des_c'); self.var_units['DTt_des_c'] = 'K'
                self.var.append('rho_te_c'); self.var_units['rho_te_c'] = 'kg/m**3'
                self.var.append('c_te_c'); self.var_units['c_te_c', ] = 'm/s'
                self.var.append('rho_ti_c'); self.var_units['rho_ti_c'] = 'kg/m**3'
                self.var.append('c_ti_c'); self.var_units['c_ti_c'] = 'm/s'

        # Fan variables
        if settings['fan_inlet_source'] or settings['fan_discharge_source']:
            self.var.append('DTt_f'); self.var_units['DTt_f'] = 'K'
            self.var.append('mdot_f'); self.var_units['mdot_f'] = 'kg/s'
            self.var.append('N_f'); self.var_units['N_f'] = 'rpm'
            self.var.append('A_f'); self.var_units['A_f'] = 'm**2'
            self.var.append('d_f'); self.var_units['d_f'] = 'm'

        return None

    def get_deck(self, settings) -> None:
        """
        Load engine deck for trajectory computations.

        :return: None
        """

        # Load self.engine data and create interpolation functions
        data = pd.read_csv(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/engine/' + settings['engine_deck_file_name'])
        data_column_labels = {'F_n':'Fn [N]',
                              'W_f':'Wf [kg/s]',
                              'V_j':'jet_V [m/s]',
                              'Tt_j':'jet_Tt [K]',
                              'rho_j':'jet_rho [kg/m3]',
                              'A_j':'jet_A [m2]',
                              'M_j':'jet_M [-]',
                              'mdoti_c':'core_mdot_in [kg/s]',
                              'Tti_c':'core_Tt_in [K]',
                              'Ttj_c':'core_Tt_out [K]',
                              'Pti_c':'core_Pt_in [Pa]',
                              'DTt_des_c':'core_DT_t [K]',
                              'rho_te_c':'core_LPT_rho_out [kg/m3]',
                              'rho_ti_c':'core_HPT_rho_in [kg/m3]',
                              'c_te_c':'core_LPT_c_out [m/s]',
                              'c_ti_c':'core_HPT_c_in [m/s]',
                              'DTt_f':'fan_DTt [K]',
                              'mdot_f':'fan_mdot_in [kg/s]',
                              'N_f':'fan_N [rpm]',
                              'A_f':'fan_A [m2]',
                              'd_f':'fan_d [m]',
                              'M_d_f':'fan_M_d [-]'}

        # Initialize engine deck variables
        if settings['atmosphere_type'] == 'stratified':
            self.deck['z'] = np.unique(data['z [m]'].values)
            self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
            self.deck['TS'] = np.unique(data['T/TMAX [-]'].values)
            
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
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = settings['F00']

                                else:
                                    if settings['thrust_lapse']:
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[1]
            else:
                raise ValueError('deck_var list is empty') 

        elif settings['atmosphere_type'] == 'sealevel':

            # Select only sealevel values in engine deck
            data = data[data['z [m]'] == 0]

            self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
            self.deck['TS'] = np.unique(data['T/TMAX [-]'].values)

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
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = settings['F00']

                            else:
                                if settings['thrust_lapse']:
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[1]

            else:
                raise ValueError('deck_var list is empty') 

        return None

    def load_timeseries_csv(self, settings) -> None:
        
        """
        Load engine timeseries from .csv file.

        :param timestep:
        :type timestep: int

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.timeseries = pd.read_csv(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/engine/' + settings['output_directory_name'] + '/' + settings['engine_timeseries_file_name'])
        
        return None

    def load_timeseries_operating_point(self, timestep):

        # Select operating point
        cols = self.timeseries.columns
        op_point = pd.DataFrame(np.reshape(self.timeseries.values[timestep, :], (1, len(cols))))
        op_point.columns = cols

        # Duplicate operating for theta range (np.linspace(0, 180, 19))
        self.timeseries = pd.DataFrame()
        for _ in np.arange(19):
            self.timeseries = self.timeseries.append(op_point)

        return None


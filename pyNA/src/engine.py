import pandas as pd
import numpy as np
import pdb


class Engine():

    def __init__(self, pyna_directory, ac_name, case_name, output_directory_name, engine_timeseries_name, engine_deck_name) -> None:
        
        # Settings 
        self.pyna_directory = pyna_directory
        self.ac_name = ac_name
        self.case_name = case_name
        self.output_directory_name = output_directory_name
        self.engine_timeseries_name = engine_timeseries_name
        self.engine_deck_name = engine_deck_name

        # Instantiate 
        self.timeseries = pd.DataFrame()
        self.deck = dict()
        self.deck_variables = dict()
        
    def get_timeseries(self, timestep=None) -> None:
        
        """
        Load engine timeseries from .csv file.

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.timeseries = pd.read_csv(self.pyna_directory + '/cases/' + self.case_name + '/engine/' + self.output_directory_name + '/' + self.engine_timeseries_name)

        if not timestep==None:
            # Select operating point
            cols = self.timeseries.columns
            op_point = pd.DataFrame(np.reshape(self.timeseries.values[timestep, :], (1, len(cols))))
            op_point.columns = cols

            # Duplicate operating for theta range (np.linspace(0, 180, 19))
            self.timeseries = pd.DataFrame()
            for i in np.arange(19):
                self.timeseries = self.timeseries.append(op_point)

        return None

    def get_performance_deck_variables(self, fan_inlet_source, fan_discharge_source, core_source, jet_mixing_source, jet_shock_source, core_turbine_attenuation_method='ge'):
        """
        Get list of engine variables required from the engine deck

        :return: None
        """

        # General variables
        self.deck_variables['F_n'] = 'N'
        self.deck_variables['W_f'] = 'kg/s'
        self.deck_variables['Tti_c'] = 'K'
        self.deck_variables['Pti_c'] = 'Pa'

        # Jet variables
        if jet_mixing_source and not jet_shock_source:
            self.deck_variables['V_j'] = 'm/s' 
            self.deck_variables['rho_j'] = 'kg/m**3'
            self.deck_variables['A_j'] = 'm**2'
            self.deck_variables['Tt_j'] = 'K'
        elif jet_shock_source and not jet_mixing_source:
            self.deck_variables['V_j'] = 'm/s' 
            self.deck_variables['A_j'] = 'm**2'
            self.deck_variables['Tt_j'] = 'K'
            self.deck_variables['M_j'] = None
        elif jet_mixing_source and jet_shock_source:
            self.deck_variables['V_j'] = 'm/s' 
            self.deck_variables['rho_j'] = 'kg/m**3'
            self.deck_variables['A_j'] = 'm**2'
            self.deck_variables['Tt_j'] = 'K'
            self.deck_variables['M_j'] = None
        
        # Core variables
        if core_source:
            if core_turbine_attenuation_method == 'ge':
                self.deck_variables['mdoti_c'] = 'kg/s'
                self.deck_variables['Ttj_c'] = 'K'
                self.deck_variables['DTt_des_c'] = 'K'

            elif core_turbine_attenuation_method.method_core_turb == 'pw':
                self.deck_variables['mdoti_c'] = 'kg/s'
                self.deck_variables['Ttj_c'] = 'K'
                self.deck_variables['DTt_des_c'] = 'K'
                self.deck_variables['rho_te_c'] = 'kg/m**3'
                self.deck_variables['c_te_c', ] = 'm/s'
                self.deck_variables['rho_ti_c'] = 'kg/m**3'
                self.deck_variables['c_ti_c'] = 'm/s'

        # Fan variables
        if fan_inlet_source or fan_discharge_source:
            self.deck_variables['DTt_f'] = 'K'
            self.deck_variables['mdot_f'] = 'kg/s'
            self.deck_variables['N_f'] = 'rpm'
            self.deck_variables['A_f'] = 'm**2'
            self.deck_variables['d_f'] = 'm'

        return None

    def get_performance_deck(self, atmosphere_type, thrust_lapse, F00=None) -> None:
        """
        Load engine deck for trajectory computations.

        :return: None
        """

        # Load self.engine data and create interpolation functions
        data = pd.read_csv(self.pyna_directory + '/cases/' + self.case_name + '/engine/' + self.engine_deck_name)
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
        if atmosphere_type == 'stratified':
            self.deck['z'] = np.unique(data['z [m]'].values)
            self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
            self.deck['TS'] = np.unique(data['T/TMAX [-]'].values)
            
            if len(self.deck_variables.keys()) == 0:
                raise ValueError('deck_variables is empty') 
            else:                
                for var in self.deck_variables:
                    self.deck[var] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

                # Fill engine deck
                cntr = -1
                for i in np.arange(self.deck['z'].shape[0]):
                    for j in np.arange(self.deck['M_0'].shape[0]):
                        for k in np.flip(np.arange(self.deck['TS'].shape[0])):
                            cntr = cntr + 1

                            for var in self.deck_variables:

                                if var == 'F_n':
                                    if thrust_lapse:
                                        # self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]/83821.6*F00
                                        # self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]/136325.9272*F00
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = F00

                                else:
                                    if thrust_lapse:
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[cntr]
                                    else:
                                        self.deck[var][i, j, k] = data[data_column_labels[var]].values[1]

        elif atmosphere_type == 'sealevel':

            # Select only sealevel values in engine deck
            data = data[data['z [m]'] == 0]

            self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
            self.deck['TS'] = np.unique(data['T/TMAX [-]'].values)

            if len(self.deck_variables.keys()) == 0:
                raise ValueError('deck_variables is empty') 
            else:
                for var in self.deck_variables:
                    self.deck[var] = np.zeros((self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

                # Fill engine deck
                cntr = -1
                for j in np.arange(self.deck['M_0'].shape[0]):
                    for k in np.flip(np.arange(self.deck['TS'].shape[0])):
                        cntr = cntr + 1

                        for var in self.deck_variables:

                            if var == 'F_n':
                                if thrust_lapse:
                                    # self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]/83821.6*F00
                                    # self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]/136325.9272*F00
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = F00

                            else:
                                if thrust_lapse:
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[cntr]
                                else:
                                    self.deck[var][j, k] = data[data_column_labels[var]].values[1]
    

        return None




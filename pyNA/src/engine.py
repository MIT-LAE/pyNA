import pdb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pyNA.src.settings import Settings


@dataclass
class Engine:
    """
    Engine class containing the following parameters

    * ``time_series``:   time series of engine parameters for a predefined trajectory
    * ``deck``:         engine deck used for trajectory computations

    """

    # Initialize aircraft name
    def __init__(self) -> None:

        # Initialization
        self.time_series = dict()
        self.deck = dict()
        self.TS_limit = dict()

    def load_time_series(self, settings, engine_file_name='Engine_to.csv') -> None:
        """
        Load engine time series for noise computations.

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param engine_file_name: File name of engine time series
        :type engine_file_name: str

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.time_series = pd.read_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/engine/' + settings.output_directory_name + '/' + engine_file_name)

        return None

    def load_operating_point(self, settings, time_step, engine_file_name='Engine_to.csv') -> None:
        """
        Load engine operating point for noise distribution calculations.

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param time_step: 
        :type time_step:
        :param engine_file_name: File name of engine time series
        :type engine_file_name: str

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.time_series = pd.read_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/engine/' + settings.output_directory_name + '/' + engine_file_name)

        # Select operating point
        cols = self.time_series.columns
        op_point = pd.DataFrame(np.reshape(self.time_series.values[time_step, :], (1, len(cols))))
        op_point.columns = cols

        # Duplicate operating for theta range (np.linspace(0, 180, 19))
        self.time_series = pd.DataFrame()
        for i in np.arange(19):
            self.time_series = self.time_series.append(op_point)

        return None

    def load_deck(self, settings: Settings) -> None:
        """
        Load engine deck for trajectory computations.

        :return: None
        """

        # Load self.engine data and create interpolation functions
        data = pd.read_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/engine/' + settings.engine_file_name)
        self.deck['z'] = np.unique(data['z [m]'].values)
        self.deck['M_0'] = np.unique(data['M_0 [-]'].values)
        self.deck['TS'] = np.unique(data['T/TMAX [-]'].values)
        self.deck['F_n'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['W_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

        self.deck['V_j'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['Tt_j'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['rho_j'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['A_j'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['M_j'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

        self.deck['mdot_i_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['Tti_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['Ttj_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['Pti_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['DTt_des_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['rho_te_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['rho_ti_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['c_te_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['c_ti_c'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

        self.deck['DTt_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['mdot_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['N_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['A_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['d_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))
        self.deck['M_d_f'] = np.zeros((self.deck['z'].shape[0], self.deck['M_0'].shape[0], self.deck['TS'].shape[0]))

        cntr = -1
        for i in np.arange(self.deck['z'].shape[0]):
            for j in np.arange(self.deck['M_0'].shape[0]):
                for k in np.flip(np.arange(self.deck['TS'].shape[0])):
                    cntr = cntr + 1

                    if settings.engine_thrust_lapse:
                        if settings.ac_name == 'stca' and not settings.Foo == None:
                            self.deck['F_n'][i, j, k] = data['Fn [N]'].values[cntr] / 83821.6 * settings.Foo
                        elif settings.ac_name == 'a10' and not settings.Foo == None:
                            self.deck['F_n'][i, j, k] = data['Fn [N]'].values[cntr] / 136325.9272 * settings.Foo
                        else:
                            self.deck['F_n'][i, j, k] = data['Fn [N]'].values[cntr]
                        self.deck['W_f'][i, j, k] = data['Wf [kg/s]'].values[cntr]
                        self.deck['V_j'][i, j, k] = data['jet_V [m/s]'].values[cntr]
                        self.deck['Tt_j'][i, j, k] = data['jet_Tt [K]'].values[cntr]
                        self.deck['rho_j'][i, j, k] = data['jet_rho [kg/m3]'].values[cntr]
                        self.deck['A_j'][i, j, k] = data['jet_A [m2]'].values[cntr]
                        self.deck['M_j'][i, j, k] = data['jet_M [-]'].values[cntr]
                        self.deck['mdot_i_c'][i, j, k] = data['core_mdot_in [kg/s]'].values[cntr]
                        self.deck['Tti_c'][i, j, k] = data['core_Tt_in [K]'].values[cntr]
                        self.deck['Ttj_c'][i, j, k] = data['core_Tt_out [K]'].values[cntr]
                        self.deck['Pti_c'][i, j, k] = data['core_Pt_in [Pa]'].values[cntr]
                        self.deck['DTt_des_c'][i, j, k] = data['core_DT_t [K]'].values[cntr]
                        self.deck['rho_te_c'][i, j, k] = data['core_LPT_rho_out [kg/m3]'].values[cntr]
                        self.deck['rho_ti_c'][i, j, k] = data['core_HPT_rho_in [kg/m3]'].values[cntr]
                        self.deck['c_te_c'][i, j, k] = data['core_LPT_c_out [m/s]'].values[cntr]
                        self.deck['c_ti_c'][i, j, k] = data['core_HPT_c_in [m/s]'].values[cntr]
                        self.deck['DTt_f'][i, j, k] = data['fan_DTt [K]'].values[cntr]
                        self.deck['mdot_f'][i, j, k] = data['fan_mdot_in [kg/s]'].values[cntr]
                        self.deck['N_f'][i, j, k] = data['fan_N [rpm]'].values[cntr]
                        self.deck['A_f'][i, j, k] = data['fan_A [m2]'].values[cntr]
                        self.deck['d_f'][i, j, k] = data['fan_d [m]'].values[cntr]
                        self.deck['M_d_f'][i, j, k] = data['fan_M_d [-]'].values[cntr]

                    else:
                        self.deck['F_n'][i, j, k] = settings.Foo
                        self.deck['W_f'][i, j, k] = data['Wf [kg/s]'].values[1]
                        self.deck['V_j'][i, j, k] = data['jet_V [m/s]'].values[1]
                        self.deck['Tt_j'][i, j, k] = data['jet_Tt [K]'].values[1]
                        self.deck['rho_j'][i, j, k] = data['jet_rho [kg/m3]'].values[1]
                        self.deck['A_j'][i, j, k] = data['jet_A [m2]'].values[1]
                        self.deck['M_j'][i, j, k] = data['jet_M [-]'].values[1]
                        self.deck['mdot_i_c'][i, j, k] = data['core_mdot_in [kg/s]'].values[1]
                        self.deck['Tti_c'][i, j, k] = data['core_Tt_in [K]'].values[1]
                        self.deck['Ttj_c'][i, j, k] = data['core_Tt_out [K]'].values[1]
                        self.deck['Pti_c'][i, j, k] = data['core_Pt_in [Pa]'].values[1]
                        self.deck['DTt_des_c'][i, j, k] = data['core_DT_t [K]'].values[1]
                        self.deck['rho_te_c'][i, j, k] = data['core_LPT_rho_out [kg/m3]'].values[1]
                        self.deck['rho_ti_c'][i, j, k] = data['core_HPT_rho_in [kg/m3]'].values[1]
                        self.deck['c_te_c'][i, j, k] = data['core_LPT_c_out [m/s]'].values[1]
                        self.deck['c_ti_c'][i, j, k] = data['core_HPT_c_in [m/s]'].values[1]
                        self.deck['DTt_f'][i, j, k] = data['fan_DTt [K]'].values[1]
                        self.deck['mdot_f'][i, j, k] = data['fan_mdot_in [kg/s]'].values[1]
                        self.deck['N_f'][i, j, k] = data['fan_N [rpm]'].values[1]
                        self.deck['A_f'][i, j, k] = data['fan_A [m2]'].values[1]
                        self.deck['d_f'][i, j, k] = data['fan_d [m]'].values[1]
                        self.deck['M_d_f'][i, j, k] = data['fan_M_d [-]'].values[1]

        return None

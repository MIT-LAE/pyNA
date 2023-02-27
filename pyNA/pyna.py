
from pyNA.src.trajectory import Trajectory
from pyNA.src.aircraft import Aircraft
import matplotlib.pyplot as plt
import pdb
import numpy as np


class pyna:

    def __init__(self, 
                case_name = 'nasa_stca_standard',
                ac_name = 'stca',
                output_directory_name = '',
                output_file_name = 'trajectory_stca.sql',
                time_history_file_name = 'time_history.csv',
                engine_deck_file_name = 'engine_deck_stca.csv',
                atmosphere_mode = 'stratified',
                fan_BB_method = 'geae',
                fan_RS_method = 'allied_signal',
                fan_ge_flight_cleanup = 'takeoff',
                core_turbine_attenuation_method = 'ge',
                trajectory_mode='data',
                lateral_attenuation_engine_mounting = 'underwing',
                levels_int_metric = 'epnl',
                observer_lst = ('lateral', 'flyover'),
                noise = False,
                emissions = False,
                thrust_lapse = True,
                ptcb = False,
                phld = False,
                pkrot = False,
                all_sources = True,
                fan_inlet_source = False,
                fan_discharge_source = False,
                core_source = False,
                jet_mixing_source = False,
                jet_shock_source = False,
                airframe_source = False,
                fan_igv = False,
                fan_id = False,
                fan_combination_tones = False,
                fan_liner_suppression = True,
                airframe_hsr_calibration = True,
                direct_propagation = True,
                absorption = True,
                ground_effects = True,
                lateral_attenuation = True,
                shielding = False,
                tones_under_800Hz = False,
                epnl_bandshare = False,
                core_jet_suppression = False,
                save_results = False,
                verification = False,
                F00 = None,
                z_cb = 500.,
                v_max = 128.6,
                x_max = 15000.,
                epnl_dt = 0.5,
                atmosphere_dT = 10.0169,
                max_iter = 200,
                tolerance = 1e-6,
                x_observer_array = np.array([[12325.*0.3048, 450., 4*0.3048], [21325.*0.3048, 0., 4*0.3048]]),
                ground_resistance = 291.0 * 515.379,
                incoherence_constant = 0.01,
                n_frequency_bands = 24,
                n_frequency_subbands = 5,
                n_altitude_absorption = 5,
                n_harmonics = 10,
                n_shock = 8,
                A_e = 10.334 * (0.3048 ** 2),
                r_0 = 0.3048,
                p_ref = 2e-5) -> None:
        
        """
        
        :param case_name:
        :type case_name: str
        :param ac_name:
        :type ac_name: str
        :param output_directory_name:
        :type output_directory_name: str
        :param output_file_name:
        :type output_file_name: str
        :param time_history_file_name:
        :type time_history_file_name: str
        :param engine_deck_file_name:
        :type engine_deck_file_name: str
        :param atmosphere_mode:
        :type atmosphere_mode: str
        :param fan_BB_method:
        :type fan_BB_method: str
        :param fan_RS_method:
        :type fan_RS_method: str
        :param fan_ge_flight_cleanup:
        :type fan_ge_flight_cleanup: str
        :param core_turbine_attenuation_method:
        :type core_turbine_attenuation_method: str
        :param trajectory_mode:
        :type trajectory_mode: str
        :param lateral_attenuation_engine_mounting:
        :type lateral_attenuation_engine_mounting: str
        :param levels_int_metric:
        :type levels_int_metric: str
        :param observer_lst:
        :type observer_lst: list
        :param noise:
        :type noise: bool
        :param emissions:
        :type emissions: bool
        :param thrust_lapse:
        :type thrust_lapse: bool
        :param ptcb:
        :type ptcb: bool
        :param phld:
        :type phld: bool
        :param pkrot:
        :type pkrot: bool
        :param all_sources:
        :type all_sources: bool
        :param fan_inlet_source:
        :type fan_inlet_source: bool
        :param fan_discharge_source:
        :type fan_discharge_source: bool
        :param core_source:
        :type core_source: bool
        :param jet_mixing_source:
        :type jet_mixing_source: bool
        :param jet_shock_source:
        :type jet_shock_source: bool
        :param airframe_source:
        :type airframe_source: bool
        :param fan_igv:
        :type fan_igv: bool
        :param fan_id:
        :type fan_id: bool
        :param fan_combination_tones:
        :type fan_combination_tones: bool
        :param fan_liner_suppression:
        :type fan_liner_suppression: bool
        :param airframe_hsr_calibration:
        :type airframe_hsr_calibration: bool
        :param direct_propagation:
        :type direct_propagation: bool
        :param absorption:
        :type absorption: bool
        :param ground_effects:
        :type ground_effects: bool
        :param lateral_attenuation:
        :type lateral_attenuation: bool
        :param shielding:
        :type shielding: bool
        :param tones_under_800Hz:
        :type tones_under_800Hz: bool
        :param epnl_bandshare:
        :type epnl_bandshare: bool
        :param core_jet_suppression:
        :type core_jet_suppression: bool
        :param save_results:
        :type save_results: bool
        :param verification:
        :type verification: bool
        :param F00:
        :type F00: float
        :param z_cb:
        :type z_cb: float
        :param v_max:
        :type v_max: float
        :param x_max:
        :type x_max: float
        :param epnl_dt:
        :type epnl_dt: float
        :param atmosphere_dT:
        :type atmosphere_dT: float
        :param max_iter:
        :type max_iter: int
        :param tolerance:
        :type tolerance: float
        :param x_observer_array:
        :type x_observer_array: np.ndarray
        :param ground_resistance:
        :type ground_resistance: float
        :param incoherence_constant:
        :type incoherence_constant: float
        :param n_frequency_bands:
        :type n_frequency_bands: int
        :param n_frequency_subbands:
        :type n_frequency_subbands: int
        :param n_altitude_absorption:
        :type n_altitude_absorption: int
        :param n_harmonics:
        :type n_harmonics: int
        :param n_shock:
        :type n_shock: int
        :param A_e:
        :type A_e: float
        :param r_0:
        :type r_0: float
        :param p_ref
        :type p_ref: float
        """

        self.settings = dict()
        self.settings['case_name'] = case_name
        self.settings['ac_name'] = ac_name
        self.settings['output_directory_name'] = output_directory_name
        self.settings['output_file_name'] = output_file_name
        self.settings['time_history_file_name'] = time_history_file_name
        self.settings['engine_deck_file_name'] = engine_deck_file_name
        self.settings['atmosphere_mode'] = atmosphere_mode
        self.settings['fan_BB_method'] = fan_BB_method
        self.settings['fan_RS_method'] = fan_RS_method
        self.settings['fan_ge_flight_cleanup'] = fan_ge_flight_cleanup
        self.settings['core_turbine_attenuation_method'] = core_turbine_attenuation_method
        self.settings['trajectory_mode'] = trajectory_mode
        self.settings['lateral_attenuation_engine_mounting'] = lateral_attenuation_engine_mounting
        self.settings['levels_int_metric'] = levels_int_metric
        self.settings['observer_lst'] = observer_lst
        self.settings['noise'] = noise
        self.settings['emissions'] = emissions
        self.settings['thrust_lapse'] = thrust_lapse
        self.settings['ptcb'] = ptcb
        self.settings['phld'] = phld
        self.settings['pkrot'] = pkrot
        self.settings['all_sources'] = all_sources
        self.settings['fan_inlet_source'] = fan_inlet_source
        self.settings['fan_discharge_source'] = fan_discharge_source
        self.settings['core_source'] = core_source
        self.settings['jet_mixing_source'] = jet_mixing_source
        self.settings['jet_shock_source'] = jet_shock_source
        self.settings['airframe_source'] = airframe_source
        self.settings['fan_igv'] = fan_igv
        self.settings['fan_id'] = fan_id
        self.settings['fan_combination_tones'] = fan_combination_tones
        self.settings['fan_liner_suppression'] = fan_liner_suppression
        self.settings['airframe_hsr_calibration'] = airframe_hsr_calibration
        self.settings['direct_propagation'] = direct_propagation
        self.settings['absorption'] = absorption
        self.settings['ground_effects'] = ground_effects
        self.settings['lateral_attenuation'] = lateral_attenuation
        self.settings['shielding'] = shielding
        self.settings['tones_under_800Hz'] = tones_under_800Hz
        self.settings['epnl_bandshare'] = epnl_bandshare
        self.settings['core_jet_suppression'] = core_jet_suppression
        self.settings['save_results'] = save_results
        self.settings['verification'] = verification
        self.settings['F00'] = F00
        self.settings['z_cb'] = z_cb
        self.settings['v_max'] = v_max
        self.settings['x_max'] = x_max
        self.settings['epnl_dt'] = epnl_dt
        self.settings['atmosphere_dT'] = atmosphere_dT
        self.settings['max_iter'] = max_iter
        self.settings['tolerance'] = tolerance
        self.settings['x_observer_array'] = x_observer_array
        self.settings['ground_resistance'] = ground_resistance
        self.settings['incoherence_constant'] = incoherence_constant
        self.settings['n_frequency_bands'] = n_frequency_bands
        self.settings['n_frequency_subbands'] = n_frequency_subbands
        self.settings['n_altitude_absorption'] = n_altitude_absorption
        self.settings['n_harmonics'] = n_harmonics
        self.settings['n_shock'] = n_shock
        self.settings['A_e'] = A_e
        self.settings['r_0'] = r_0
        self.settings['p_ref'] = p_ref

        self.aircraft = Aircraft(settings=self.settings)
        if self.settings['trajectory_mode'] == 'model':
            self.aircraft.get_aerodynamics_deck(settings=self.settings)
            self.aircraft.engine.get_performance_deck(settings=self.settings)

        self.trajectory = Trajectory(settings=self.settings)

    def plot_ipopt_convergence_data():
        pass

    def plot_trajectory(self, paths_compare=[], labels_compare=[]):

        """
        
        :param paths_compare:
        :type paths_compare: tuple of om.Problem()
        """

        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use('plot.mplstyle')

        ax[0,0].plot(self.trajectory.path.get_val('trajectory.x'), self.trajectory.path.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        for i,path in enumerate(paths_compare):
            ax[0,0].plot(path.get_val('trajectory.x'), path.get_val('trajectory.z'), '-', label=labels_compare[i])
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.v'), '-', color='k')
        for path in paths_compare:
            ax[0,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.v'), '-')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.gamma'), '-', color='k')
        for path in paths_compare:
            ax[0,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.gamma'), '-')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.trajectory.path.get_val('trajectory.t_s'), 1 / 1000. * self.trajectory.path.get_val('trajectory.F_n'), '-', color='k')
        for path in paths_compare:
            ax[1,0].plot(path.get_val('trajectory.t_s'), 1 / 1000. * path.get_val('trajectory.F_n'), '-')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.tau'), '-', color='k')
        for path in paths_compare:
            ax[1,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.tau'), '-')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$\tau$ [-]')
        ax[1,1].set_ylim([0,1.02])
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.alpha'), '-', color='k')
        for path in paths_compare:
            ax[1,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.alpha'), '-')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None


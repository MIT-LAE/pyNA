from pyNA.src.airframe import Airframe
from pyNA.src.engine import Engine
from pyNA.src.data import Data
from pyNA.src.trajectory import Trajectory
from pyNA.src.noise_timeseries import NoiseTimeSeries
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import openmdao.api as om
import pandas as pd
import os
import copy
import pdb


class pyna:

    def __init__(self,
                case_name = 'nasa_stca_standard',
                ac_name = 'stca',
                pyna_directory = '/Users/laurensvoet/Documents/Research/pyNA/pyNA',
                output_directory_name = '',
                output_file_name = 'trajectory_stca.sql',
                engine_timeseries_name = 'Engine_to.csv',
                engine_deck_name = 'engine_deck_stca.csv',
                trajectory_file_name = 'Trajectory_to.csv',
                thrust_lapse = True,
                atmosphere_type = 'stratified',
                atmosphere_dT = 10.0169,
                ptcb = False,
                phld = False,
                pkrot = False,
                TS_to = 1.0,
                TS_vnrs = 1.0,
                TS_cb = 1.0,
                z_cb = 500.,
                v_max = 128.6,
                k_rot = 1.3,
                theta_flaps = 10.,
                theta_slats = -6.,
                max_iter = 200,
                tolerance = 1e-6,
                F00 = None,
                noise_optimization = False,
                noise_constraint_lateral = 200.,
                fan_inlet_source = False,
                fan_discharge_source = False,
                core_source = False,
                jet_mixing_source = False,
                jet_shock_source = False,
                airframe_source = False,
                all_sources = True,
                fan_igv = False,
                fan_id = False,
                fan_BB_method = 'geae',
                fan_RS_method = 'allied_signal',
                fan_ge_flight_cleanup = 'takeoff',
                fan_combination_tones = False,
                fan_liner_suppression = True,
                core_turbine_attenuation_method = 'ge',
                airframe_hsr_calibration = True,
                direct_propagation = True,
                absorption = True,
                ground_effects = True,
                lateral_attenuation = True,
                shielding = False,
                lateral_attenuation_engine_mounting = 'underwing',
                tones_under_800Hz = False,
                levels_int_metric = 'epnl',
                epnl_bandshare = False,
                core_jet_suppression = True,
                observer_lst = ('lateral', 'flyover'),
                x_observer_array = np.array([[12325.*0.3048, 450., 4*0.3048], [21325.*0.3048, 0., 4*0.3048]]),
                ground_resistance = 291.0 * 515.379,
                incoherence_constant = 0.01,
                n_frequency_bands = 24,
                n_frequency_subbands = 5 ,
                n_altitude_absorption = 5,
                n_harmonics = 10,
                n_shock = 8,
                A_e = 10.334 * (0.3048 ** 2),
                dt_epnl = 0.5,
                r_0 = 0.3048,
                p_ref = 2e-5,
                save_results = False,
                verification = False) -> None:
        
        # File, direcotories
        self.case_name = case_name
        self.ac_name = ac_name
        self.language = os.environ['pyna_language']
        self.pyna_directory = pyna_directory
        self.output_directory_name = output_directory_name
        self.output_file_name = output_file_name
        self.engine_timeseries_name = engine_timeseries_name
        self.engine_deck_name = engine_deck_name
        self.trajectory_file_name = trajectory_file_name
        
        # Trajectory
        self.thrust_lapse = thrust_lapse
        self.atmosphere_type = atmosphere_type
        self.atmosphere_dT = atmosphere_dT
        self.ptcb = ptcb
        self.phld = phld
        self.pkrot = pkrot
        self.TS_to = TS_to
        self.TS_vnrs = TS_vnrs
        self.TS_cb = TS_cb
        self.z_cb = z_cb
        self.v_max = v_max
        self.k_rot = k_rot
        self.theta_flaps = theta_flaps
        self.theta_slats = theta_slats
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.F00 = F00
        self.noise_optimization = noise_optimization
        self.noise_constraint_lateral = noise_constraint_lateral
        
        # Noise
        self.fan_inlet_source = fan_inlet_source
        self.fan_discharge_source = fan_discharge_source
        self.core_source = core_source
        self.jet_mixing_source = jet_mixing_source
        self.jet_shock_source = jet_shock_source
        self.airframe_source = airframe_source
        self.all_sources = all_sources
        self.fan_igv = fan_igv
        self.fan_id = fan_id
        self.fan_BB_method = fan_BB_method
        self.fan_RS_method = fan_RS_method
        self.fan_ge_flight_cleanup = fan_ge_flight_cleanup
        self.fan_combination_tones = fan_combination_tones
        self.fan_liner_suppression = fan_liner_suppression
        self.core_turbine_attenuation_method = core_turbine_attenuation_method
        self.airframe_hsr_calibration = airframe_hsr_calibration
        self.direct_propagation = direct_propagation
        self.absorption = absorption
        self.ground_effects = ground_effects
        self.lateral_attenuation = lateral_attenuation
        self.shielding = shielding
        self.lateral_attenuation_engine_mounting = lateral_attenuation_engine_mounting
        self.tones_under_800Hz = tones_under_800Hz
        self.levels_int_metric = levels_int_metric
        self.epnl_bandshare = epnl_bandshare
        self.core_jet_suppression = core_jet_suppression
        self.observer_lst = observer_lst
        self.x_observer_array = x_observer_array
        self.ground_resistance = ground_resistance
        self.incoherence_constant = incoherence_constant
        self.n_frequency_bands = n_frequency_bands
        self.n_frequency_subbands = n_frequency_subbands
        self.n_altitude_absorption = n_altitude_absorption
        self.n_harmonics = n_harmonics
        self.n_shock = n_shock
        self.A_e = A_e
        self.dt_epnl = dt_epnl
        self.r_0 = r_0
        self.p_ref = p_ref

        # Post-processing
        self.save_results = save_results
        self.verification = verification

    def initialize(self) -> None:
        
        """
        Initialize pyna by defining: 

        * sealevel atmospheric properties
        * airframe
        * engine
        * noise settings
        * noise data 
        """

        self.sealevel_atmosphere = dict()
        self.sealevel_atmosphere['g'] = 9.80665
        self.sealevel_atmosphere['R'] = 287.05
        self.sealevel_atmosphere['T_0'] = 288.15
        self.sealevel_atmosphere['c_0'] = 340.294
        self.sealevel_atmosphere['p_0'] = 101325.
        self.sealevel_atmosphere['rho_0'] = 1.225
        self.sealevel_atmosphere['mu_0'] = 1.7894e-5
        self.sealevel_atmosphere['k_0'] = 25.5e-3
        self.sealevel_atmosphere['lapse_0'] = 0.0065
        self.sealevel_atmosphere['gamma'] = 1.4
        self.sealevel_atmosphere['rh_0'] = 70.
        self.sealevel_atmosphere['I_0'] = 409.74

        # Set all noise components equal to True if settings.all_sources == True
        if self.all_sources:
            self.fan_inlet_source = True
            self.fan_discharge_source = True
            self.core_source = True
            self.jet_mixing_source = True
            self.jet_shock_source = True
            self.airframe_source = True

        # Set lateral and flyover observer locations for nasa_stca_standard trajectory
        if self.case_name in ['nasa_stca_standard', 'stca_enginedesign_standard']:
            if self.observer_lst == 'lateral':
                self.x_observer_array = np.array([[3756.66, 450., 1.2192]])

            elif self.observer_lst == 'flyover':
                self.x_observer_array = np.array([[6500., 0., 1.2192]])

            elif self.observer_lst == ['lateral', 'flyover'] or self.observer_lst == ['flyover', 'lateral']:
                self.x_observer_array = np.array([[3756.66, 450., 1.2192], [6500., 0., 1.2192]])

        # Disable validation if not nasa_stca_standard trajectory
        if not self.case_name == 'nasa_stca_standard':
            self.verification = False

        # Initialize airframe, engine, noise data
        self.airframe = Airframe(pyna_directory=self.pyna_directory, ac_name=self.ac_name, case_name=self.case_name)

        self.engine = Engine(pyna_directory=self.pyna_directory, ac_name=self.ac_name, case_name=self.case_name, output_directory_name=self.output_directory_name, 
                             engine_timeseries_name=self.engine_timeseries_name, engine_deck_name=self.engine_deck_name)

        self.noise_settings = pyna.get_noise_settings(self)
        self.noise_data = Data(pyna_directory=self.pyna_directory, case_name=self.case_name, settings=self.noise_settings)

        return None

    def load_pyna_config(self) -> None:

        return None

    def load_path_timeseries(self, timestep=None) -> None:
        """
        Loads predefined trajectory timeseries.

        :param timestep: Time step in predefined trajectory at which to compute the noise source distribution.
        :type timestep: np.int64

        :return: None

        """

        # Load trajectory data for the specific observer
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.path = pd.read_csv(self.pyna_directory + '/cases/' + self.case_name + '/trajectory/' + self.output_directory_name + '/' + self.trajectory_file_name)
        
        if timestep == None:
            self.n_t = np.size(self.path['t_source [s]'])
        else:
            # Select operating point
            cols = self.path.columns
            op_point = pd.DataFrame(np.reshape(self.path.values[time_step, :], (1, len(cols))))
            op_point.columns = cols

            # Duplicate operating for theta range (np.linspace(0, 180, 19))
            self.path = pd.DataFrame()
            for i in np.arange(19):
                self.path = self.path.append(op_point)

            self.n_t = 19

        return None

    def get_noise_settings(self):

        """
        Get noise settings dictionary
        """

        settings = dict()
        settings['pyna_directory'] = self.pyna_directory
        settings['case_name'] = self.case_name
        settings['output_directory_name'] = self.output_directory_name
        settings['output_file_name'] = self.output_file_name
        settings['fan_inlet_source'] = self.fan_inlet_source
        settings['fan_discharge_source'] = self.fan_discharge_source
        settings['core_source'] = self.core_source
        settings['jet_mixing_source'] = self.jet_mixing_source
        settings['jet_shock_source'] = self.jet_shock_source
        settings['airframe_source'] = self.airframe_source
        settings['all_sources'] = self.all_sources
        settings['fan_igv'] = self.fan_igv
        settings['fan_id'] = self.fan_id
        settings['fan_BB_method'] = self.fan_BB_method
        settings['fan_RS_method'] = self.fan_RS_method
        settings['fan_ge_flight_cleanup'] = self.fan_ge_flight_cleanup
        settings['fan_combination_tones'] = self.fan_combination_tones
        settings['fan_liner_suppression'] = self.fan_liner_suppression
        settings['core_turbine_attenuation_method'] = self.core_turbine_attenuation_method
        settings['airframe_hsr_calibration'] = self.airframe_hsr_calibration
        settings['direct_propagation'] = self.direct_propagation
        settings['absorption'] = self.absorption
        settings['ground_effects'] = self.ground_effects
        settings['lateral_attenuation'] = self.lateral_attenuation
        settings['shielding'] = self.shielding
        settings['lateral_attenuation_engine_mounting'] = self.lateral_attenuation_engine_mounting
        settings['tones_under_800Hz'] = self.tones_under_800Hz
        settings['levels_int_metric'] = self.levels_int_metric
        settings['epnl_bandshare'] = self.epnl_bandshare
        settings['core_jet_suppression'] = self.core_jet_suppression
        settings['observer_lst'] = self.observer_lst
        settings['x_observer_array'] = self.x_observer_array
        settings['ground_resistance'] = self.ground_resistance
        settings['incoherence_constant'] = self.incoherence_constant
        settings['n_frequency_bands'] = self.n_frequency_bands
        settings['n_frequency_subbands'] = self.n_frequency_subbands
        settings['n_altitude_absorption'] = self.n_altitude_absorption
        settings['n_harmonics'] = self.n_harmonics
        settings['n_shock'] = self.n_shock
        settings['A_e'] = self.A_e
        settings['dt_epnl'] = self.dt_epnl
        settings['r_0'] = self.r_0
        settings['p_ref'] = self.p_ref
        settings['verification'] = self.verification

        return settings

    # --- Noise-only methods --------------------------------------------------------------------------------
    def compute_noise_timeseries(self) -> None:
        """
        Compute noise for a predefined trajectory from .csv files.

        :return: None
        """

        pyna.initialize(self)

        self.engine.get_timeseries()
        pyna.load_path_timeseries(self)

        self.noise_timeseries = NoiseTimeSeries(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, save_results=self.save_results, settings=self.noise_settings, data=self.noise_data, coloring_dir=self.pyna_directory + '/cases/' + self.case_name + '/coloring_files/')
        self.noise_timeseries.create(airframe=self.airframe, n_t=self.n_t, mode='trajectory', objective='timeseries')
        self.noise_timeseries.solve(path=self.path, engine=self.engine.timeseries, mode='trajectory')

        return None

    def compute_noise_epnl_table(self) -> pd.DataFrame:

        """
        Compute table of epnl for individual noise sources and observers.

        :return: None
        """

        self.epnl_table = pd.DataFrame(epnl_table)

        self.levels_int_metric = 'epnl'

        epnl_table = np.zeros((6, len(self.observer_lst)+1))

        components = ['fan_inlet', 'fan_discharge', 'core', 'jet_mixing', 'airframe', 'all_sources']
        
        for i, comp in enumerate(components):

            # Reset component flags
            self.fan_inlet_source = False
            self.fan_discharge_source = False
            self.core_source = False
            self.jet_mixing_source = False
            self.jet_shock_source = False
            self.airframe_source = False
            self.all_sources = False

            # Enable component flag
            if comp == 'fan_inlet':
                self.fan_inlet_source = True
            elif comp == 'fan_discharge':
                self.fan_discharge_source = True
            elif comp == 'core':
                self.core_source = True
            elif comp == 'jet_mixing':
                self.jet_mixing_source = True
            elif comp == 'jet_shock':
                self.jet_shock_source = True
            elif comp == 'airframe':
                self.airframe_source = True
            elif comp == 'all_sources':
                self.fan_inlet_source = True
                self.fan_discharge_source = True
                self.core_source = True
                self.jet_mixing_source = True
                self.jet_shock_source = True
                self.airframe_source = True

            # Run pyNA
            pyna.initialize(self)
            self.engine.get_timeseries()
            pyna.load_path_timeseries(self)
            self.noise_timeseries = NoiseTimeSeries(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, save_results=self.save_results, settings=self.noise_settings, data=self.noise_data, coloring_dir=self.pyna_directory + '/cases/' + self.case_name + '/coloring_files/')
            self.noise_timeseries.create(airframe=self.airframe, n_t=self.n_t, mode='trajectory', objective='timeseries')
            self.noise_timeseries.solve(path=self.path, engine=self.engine.timeseries, mode='trajectory')

            # Save solutions
            for j in np.arange(len(self.observer_lst)):
                epnl_table[i, j] = np.round(self.problem.get_val('noise.epnl')[j], 1)
            epnl_table[i, j+1] = np.round(np.sum(self.problem.get_val('noise.epnl')), 1)

        # Create data frame for solutions
        observer_lst = list(self.observer_lst) + ['take-off']
        self.epnl_table.columns = observer_lst
        self.epnl_table.index = components

        return self.epnl_table

    def compute_noise_contours(self, x_lst: np.ndarray, y_lst: np.ndarray) -> None:
        """
        Compute noise contours for a predefined trajectory.

        :param x_lst: List of x-location of the microphones.
        :type x_lst: list
        :param y_lst: List of y-location of the microphones.
        :type y_lst: list

        :return: None
        """

        pyna.initialize(self)
        self.contours = dict()

        self.engine.get_timeseries()
        pyna.load_path_timeseries(self)

        # Get list of observers
        self.x_observer_array = np.zeros((np.size(x_lst)*np.size(y_lst), 3))
        cntr = -1
        for i, y in enumerate(y_lst):
            for j, x in enumerate(x_lst):
                cntr = cntr + 1
                self.x_observer_array[cntr, 0] = x
                self.x_observer_array[cntr, 1] = y
                self.x_observer_array[cntr, 2] = 4*0.3048

        self.noise_timeseries = NoiseTimeSeries(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, save_results=self.save_results, settings=self.noise_settings, data=self.noise_data, coloring_dir=self.pyna_directory + '/cases/' + self.case_name + '/coloring_files/')
        self.noise_timeseries.create(airframe=self.airframe, n_t=self.n_t, mode='trajectory', objective='timeseries')
        self.noise_timeseries.solve(path=self.path, engine=self.engine.timeseries, mode='trajectory')

        # Extract the contour
        self.contours['x_lst'] = x_lst
        self.contours['y_lst'] = np.hstack((-np.flip(y_lst[1:]), y_lst))
        self.contours['X'], self.contours['Y'] = np.meshgrid(self.contours['x_lst'], self.contours['y_lst'])
        
        # Initialize contour solution matrix
        contours = np.zeros((np.size(y_lst), np.size(x_lst)))
        cntr = -1
        for i, y in enumerate(y_lst):
            for j, x in enumerate(x_lst):
                cntr = cntr + 1
                contours[i,j] = self.noise_timeseries.get_val('noise.'+self.levels_int_metric)[cntr]

        # Flip epnl solution matrix for negative y-values
        self.contours[self.levels_int_metric] = np.vstack((np.flipud(contours[1:, :]), contours))

        return None

    def compute_noise_source_distribution(self, timestep=0) -> None:
        """
        Compute noise source spectral and directional distribution.

        :param timestep: Time step in predefined trajectory at which to compute the noise source distribution.
        :type timestep: np.int64

        :return: None
        """

        # Only implemented for python
        if self.language == 'julia':
            raise ValueError('This method has not yet been implemented in Julia. Set os.environ["language"]="python" and run again.')

        # Single observer setting; disable shielding
        self.shielding = False

        pyna.initialize(self)

        self.engine.get_timeseries(self, timestep=timestep)
        pyna.load_path_timeseries(self, timestep=timestep)

        self.noise_timeseries = NoiseTimeSeries(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, save_results=self.save_results, settings=self.noise_settings, data=self.noise_data, coloring_dir=self.pyna_directory + '/cases/' + self.case_name + '/coloring_files/')
        self.noise_timeseries.create(airframe=self.airframe, n_t=self.n_t, mode='distribution', objective=None)
        self.noise_timeseries.solve(path=self.path, engine=self.engine.timeseries, mode='distribution')

        return None

    # --- Trajectory-only methods ---------------------------------------------------------------------------
    def compute_trajectory(self, trajectory_mode='cutback', objective='t_end') -> bool:
        """
        Compute aircraft take-off trajectory.

        :param trajectory_mode:
        :type trajectory_mode:
        :param objective:
        :type objective:

        :return: converged
        :rtype: bool 

        """

        pyna.initialize(self)

        self.airframe.get_aerodynamics_deck()

        self.engine.get_performance_deck_variables(fan_inlet_source=self.fan_inlet_source, fan_discharge_source=self.fan_discharge_source, core_source=self.core_source, jet_mixing_source=self.jet_mixing_source, jet_shock_source=self.jet_shock_source)
        self.engine.get_performance_deck(atmosphere_type=self.atmosphere_type, thrust_lapse=self.thrust_lapse, F00=self.F00)

        self.path = Trajectory(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, output_directory_name=self.output_directory_name, output_file_name=self.output_file_name)
        self.path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, phld=self.phld, objective=objective, trajectory_mode=trajectory_mode)
        self.path.set_objective(objective='t_end')
        self.path.set_ipopt_settings(objective=objective, tolerance=self.tolerance, max_iter=self.max_iter)
        self.path.setup(force_alloc_complex=True)
        self.path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=None, trajectory_mode=trajectory_mode)
        self.path.solve(run_driver=True, save_results=self.save_results)

        # Check convergence
        converged = self.path.check_convergence(filename='IPOPT_trajectory_convergence.out')

        return converged

    # --- Noise-trajectory methods --------------------------------------------------------------------------
    def compute_trajectory_noise(self, objective='t_end', trajectory_mode='cutback', initialization_path_name=None) -> None:
        """
        Compute aircraft take-off trajectory and noise.
        
        :param objective: optimization objective
        :type objective: str
        :param trajectory_mode:
        :type trajectory_mode:
        :param initialization_path_name: Name of initialization trajectory (in output folder of case).
        :type initialization_path_name: str

        :return: None
        """

        pyna.initialize(self)

        self.airframe.get_aerodynamics_deck()

        self.engine.get_performance_deck_variables(fan_inlet_source=self.fan_inlet_source, fan_discharge_source=self.fan_discharge_source, core_source=self.core_source, 
                                                   jet_mixing_source=self.jet_mixing_source, jet_shock_source=self.jet_shock_source)
        self.engine.get_performance_deck(atmosphere_type=self.atmosphere_type, thrust_lapse=self.thrust_lapse, F00=self.F00)

        if not initialization_path_name:
            self.initialization_path = Trajectory(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, output_directory_name=self.output_directory_name, output_file_name=self.output_file_name)
            self.initialization_path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, phld=self.phld, objective=objective, trajectory_mode=trajectory_mode)
            self.initialization_path.set_objective(objective='t_end')
            self.initialization_path.set_ipopt_settings(objective=objective, tolerance=self.tolerance, max_iter=self.max_iter)
            self.initialization_path.setup(force_alloc_complex=True)
            self.initialization_path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=None, trajectory_mode=trajectory_mode)
            self.initialization_path.solve(run_driver=True, save_results=self.save_results)

            converged = self.path.check_convergence(filename='IPOPT_trajectory_convergence.out')

        else:
            self.initialization_path = pyna.load_results(self, initialization_path_name, 'final')
            converged = True

        # Check convergence
        converged = self.initialization_path.check_convergence(filename='IPOPT_trajectory_convergence.out')

        if converged:

            self.n_t = np.size(self.initialization_path.get_val('trajectory.t_s'))

            self.path = Trajectory(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, output_directory_name=self.output_directory_name, output_file_name=self.output_file_name)
            self.path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, phld=self.phld, objective=objective, trajectory_mode=trajectory_mode)
            self.path.create_noise(settings=self.noise_settings, data=self.noise_data, airframe=self.airframe, n_t=self.n_t, mode='trajectory', objective='timeseries')            
            
            self.path.set_objective(objective='t_end')
            self.path.set_ipopt_settings(objective=objective, tolerance=self.tolerance, max_iter=self.max_iter)
            self.path.setup(force_alloc_complex=True)
            self.path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=self.initialization_path, trajectory_mode=trajectory_mode)

            self.path.solve(run_driver=False, save_results=self.save_results)
            
        else:
            print('Trajectory did not converge.')

        return None

    def optimize_trajectory_noise(self, x_lateral_observer, initialization_path_name=None) -> None:
        """
        Optimize aircraft take-off trajectory for minimum noise signature.

        :param x_lateral_observer: x positions of sideline microphones to use in the control optimization
        :type x_lateral_observer: np.ndarray
        :param initialization_path_name: Name of initialization trajectory (in output folder of case).
        :type initialization_path_name: str

        :return: None
        """

        pyna.initialize(self)

        self.airframe.get_aerodynamics_deck()

        self.engine.get_performance_deck_variables(fan_inlet_source=self.fan_inlet_source, fan_discharge_source=self.fan_discharge_source, core_source=self.core_source, 
                                                   jet_mixing_source=self.jet_mixing_source, jet_shock_source=self.jet_shock_source)
        self.engine.get_performance_deck(atmosphere_type=self.atmosphere_type, thrust_lapse=self.thrust_lapse, F00=self.F00)

        # Initialization trajectory: minimize t_end
        if not initialization_path_name:
            self.initialization_path = Trajectory(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, output_directory_name=self.output_directory_name, output_file_name=self.output_file_name)
            self.initialization_path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, objective='t_end', trajectory_mode='flyover')
            self.initialization_path.set_objective(objective='t_end')
            self.initialization_path.set_ipopt_settings(objective='t_end', tolerance=self.tolerance, max_iter=self.max_iter)
            self.initialization_path.setup(force_alloc_complex=True)
            self.initialization_path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=None, trajectory_mode='flyover')
            self.initialization_path.solve(run_driver=True, save_results=self.save_results)
        else:
            self.initialization_path = pyna.load_results(self, initialization_path_name, 'final')

        # Get list of observers
        n_sideline = np.size(x_lateral_observer)
        self.noise_settings['x_observer_array'] = np.zeros((n_sideline+1, 3))
        self.noise_settings['x_observer_array'][:-1, 0] = x_lateral_observer
        self.noise_settings['x_observer_array'][:-1, 1] = 450.
        self.noise_settings['x_observer_array'][:-1, 2] = 4 * 0.3048
        self.noise_settings['x_observer_array'][-1, 0] = 6500.
        self.noise_settings['x_observer_array'][-1, 1] = 0.
        self.noise_settings['x_observer_array'][-1, 2] = 4 * 0.3048
        
        # Setup trajectory for noise computations
        self.n_t = np.size(self.initialization_path.get_val('trajectory.t_s'))

        self.path = Trajectory(pyna_directory=self.pyna_directory, case_name=self.case_name, language=self.language, output_directory_name=self.output_directory_name, output_file_name=self.output_file_name)
        self.path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, phld=self.phld, objective='noise', trajectory_mode='flyover')
        self.path.create_noise(settings=self.noise_settings, data=self.noise_data, airframe=self.airframe, n_t=self.n_t, mode='trajectory', objective='noise')            
        
        self.path.set_objective(objective='noise', noise_constraint_lateral=self.noise_constraint_lateral)
        self.path.set_ipopt_settings(objective='noise', tolerance=self.tolerance, max_iter=self.max_iter)
        self.path.setup(force_alloc_complex=True)
        self.path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=self.initialization_path, trajectory_mode='flyover')

        self.path.solve(run_driver=True, save_results=self.save_results)
        
        return None

    # --- Post-processing methods ---------------------------------------------------------------------------
    def plot_noise_timeseries(self, metric='pnlt') -> None:

        """
        Plot the noise metric along the trajectory.

        :param metric: noise metric to plot. Specify 'pnlt' or 'oaspl'/
        :type metric: str

        :return: None
        """

        # Create figure
        fig, ax = plt.subplots(1, len(self.noise_settings['observer_lst']), figsize=(20, 4.3), dpi=100)
        if len(self.noise_settings['observer_lst']) == 1:
            ax = [ax]
        ax_zoom = copy.copy(ax)
        plt.style.use(self.pyna_directory + '/utils/' + 'plot.mplstyle')

        colors = plt.cm.magma(np.linspace(0,0.8,2))

        # Iterate over observer locations
        if self.language == 'python':
            for i, observer in enumerate(self.noise_settings['observer_lst']):

                # Time range of epnl domain of dependence
                time_epnl = self.noise_timeseries.get_val('noise.t_o')[i,:][np.where(self.noise_timeseries.get_val('noise.pnlt')[i,:] > max(self.noise_timeseries.get_val('noise.pnlt')[i,:]) - 10.)]

                # Plot noise levels
                if metric == 'pnlt':
                    # Plot noise levels
                    if observer == 'lateral':
                        ax[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:180], self.noise_timeseries.get_val('noise.pnlt')[i,:180], linewidth=2.5, label='pyNA', color=colors[0])
                    else:
                        ax[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:], self.noise_timeseries.get_val('noise.pnlt')[i,:], linewidth=2.5, label='pyNA', color=colors[0])
                    ax[i].fill_between([time_epnl[0], time_epnl[-1]], [-5, -5], [1.05*np.max(self.noise_timeseries.get_val('noise.pnlt')[i,:]), 1.05*np.max(problem.get_val('noise.pnlt')[i,:])], alpha=0.15,
                                    label='EPNL domain of dependence', color=colors[0])
                    if self.noise_settings['verification']:
                        self.data.load_trajectory_verification_data(settings=self.noise_settings)
                        ax[i].plot(self.data.verification_trajectory[observer]['t observer [s]'],
                                self.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, label='NASA STCA (Berton et al.)', color=colors[1])

                    ax[i].grid(True)
                    ax[i].set_xlabel('Time after brake release [s]')
                    ax[i].tick_params(axis='both')
                    ax[i].set_ylim([-5, np.max(self.noise_timeseries.get_val('noise.pnlt')[i,:])*1.05])

                    # Zoomed-in plots
                    ax_zoom[i] = zoomed_inset_axes(ax[i], zoom=4, loc='lower right')
                    ax_zoom[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:180], self.noise_timeseries.get_val('noise.pnlt')[i,:180], linewidth=2.5, color=colors[0])
                    if self.noise_settings['verification']:
                        ax_zoom[i].plot(self.data.verification_trajectory[observer]['t observer [s]'], self.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, color=colors[1])
                    ax_zoom[i].set_xticks([])
                    ax_zoom[i].set_yticks([])
                    ax_zoom[i].set_xlim([time_epnl[0], time_epnl[-1]])
                    ax_zoom[i].set_ylim([np.max(self.noise_timeseries.get_val('noise.pnlt')[i,:])-11, np.max(self.noise_timeseries.get_val('noise.pnlt')[i,:])+1.5])
                    mark_inset(ax[i], ax_zoom[i], loc1=1, loc2=3)                

                elif metric == 'oaspl':
                    ax[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:], self.noise_timeseries.get_val('noise.oaspl')[i,:], linewidth=2.5, label='pyNA')
                    ax[i].set_ylabel('$OASPL_{' + observer + '}$ [dB]')
                    if self.noise_settings['verification']:
                        self.data.load_trajectory_verification_data(settings=self.noise_settings)
                        ax[i].plot(self.data.verification_trajectory[observer]['t observer [s]'],
                                self.data.verification_trajectory[observer]['OASPL'], '--', linewidth=2.5,
                                label='NASA STCA (Berton et al.)')

                ax[i].grid(True)
                ax[i].set_xlabel('Time after brake release [s]')
                ax[i].tick_params(axis='both')
                ax[i].set_ylim([-5, 110])

        elif self.language == 'julia':
            # Iterate over observer locations
            for i, observer in enumerate(self.noise_settings['observer_lst']):

                # Time range of epnl domain of dependence
                time_epnl = self.noise_timeseries.get_val('noise.t_o')[i,:][np.where(self.noise_timeseries.get_val('noise.level')[i,:] > max(self.noise_timeseries.get_val('noise.level')[i,:]) - 10.)]

                # Plot noise levels
                if observer == 'lateral':
                    ax[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:180], self.noise_timeseries.get_val('noise.level')[i,:180], linewidth=2.5, label='pyNA', color=colors[0])
                else:
                    ax[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:], self.noise_timeseries.get_val('noise.level')[i,:], linewidth=2.5, label='pyNA', color=colors[0])
                ax[i].fill_between([time_epnl[0], time_epnl[-1]], [-5, -5], [1.05*np.max(self.noise_timeseries.get_val('noise.level')[i,:]), 1.05*np.max(self.noise_timeseries.get_val('noise.level')[i,:])], alpha=0.15, label='EPNL domain of dependence', color=colors[0])
                if self.noise_settings['verification']:
                    self.data.load_trajectory_verification_data(settings=self.noise_settings)
                    ax[i].plot(self.data.verification_trajectory[observer]['t observer [s]'],
                            self.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, label='NASA STCA (Berton et al.)', color=colors[1])

                ax[i].grid(True)
                ax[i].set_xlabel('Time after brake release [s]')
                ax[i].tick_params(axis='both')
                ax[i].set_ylim([-5, 105])

                # Zoomed-in plots
                ax_zoom[i] = zoomed_inset_axes(ax[i], zoom=4, loc='lower right')
                ax_zoom[i].plot(self.noise_timeseries.get_val('noise.t_o')[i,:180], self.noise_timeseries.get_val('noise.level')[i,:180], linewidth=2.5, color=colors[0])
                if self.noise_settings['verification']:
                    ax_zoom[i].plot(self.data.verification_trajectory[observer]['t observer [s]'], self.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, color=colors[1])
                ax_zoom[i].set_xticks([])
                ax_zoom[i].set_yticks([])
                ax_zoom[i].set_xlim([time_epnl[0], time_epnl[-1]])
                ax_zoom[i].set_ylim([np.max(self.noise_timeseries.get_val('noise.level')[i,:])-11, np.max(self.noise_timeseries.get_val('noise.level')[i,:])+1.5])
                mark_inset(ax[i], ax_zoom[i], loc1=1, loc2=3)                

                ax[i].grid(True)
                ax[i].set_xlabel('Time after brake release [s]')
                ax[i].tick_params(axis='both')
                ax[i].set_ylim([-5, 1.05*np.max(self.noise_timeseries.get_val('noise.level')[i,:])])

        if self.noise_settings['observer_lst'] == ('lateral', 'flyover',):
            ax[0].set_title('Lateral')
            ax[1].set_title('Flyover')
        elif self.noise_settings['observer_lst'] == ('approach',):
            ax[0].set_title('Approach')
        ax[0].set_ylabel('$PNLT$ [TPNdB]')

        # Set legend
        ax[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.09), ncol=1, borderaxespad=0, frameon=False)
        plt.subplots_adjust(wspace=0.1)
        plt.show()

        return None

    def plot_noise_contours(self) -> None:
        """
        Plot noise contours around take-off trajectory.

        :return: None
        """

        # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and then adds a percent sign.
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

        plt.figure(figsize=(15, 5))
        plt.style.use(self.pyna_directory + '/utils/' + 'plot.mplstyle')

        CS = plt.contour(self.contours['X'], self.contours['Y'], self.contours['epnl'], levels=[60, 65, 70, 75, 80, 85, 90, 100, 110])
        plt.plot(self.path['X [m]'], self.path['Y [m]'], 'k-', linewidth=4, label='Trajectory')

        plt.clabel(CS, [60, 65, 70, 75, 80, 85, 90, 100, 110], inline=True, fmt=fmt, fontsize=18)
        plt.xlim([np.min(self.contours['X']), np.max(self.contours['X'])])
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.legend()
        plt.show()

        return None

    def plot_noise_source_distribution(self, metric:str, components=['core', 'jet_mixing', 'airframe', 'fan_inlet', 'fan_discharge'], timestep=1) -> None:
        
        """
        
        :param metric: 
        :type metric: str
        :param components:
        :type components: lst

        """

        # Plot noise hemispheres 
        if metric == 'spl':
            fig, ax = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(10, 10), dpi=100)
            plt.subplots_adjust(wspace=-0.,hspace=-0.2)
        elif metric == 'oaspl':
            fig, ax = plt.subplots(2, 3, figsize=(20, 8))
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
        else:
            raise ValueError('Invalid metric specified. Specify: spl / oaspl.')
        plt.style.use(self.pyna_directory + '/utils/' + 'plot.mplstyle')

        # Loop through different components
        irow = -1
        icol = -1
        for i, comp in enumerate([components]):

            self.all_sources = False
            self.fan_inlet_source = False
            self.fan_discharge_source = False
            self.core_source = False
            self.jet_mixing_source = False
            self.jet_shock_source = False
            self.airframe_source = False

            if i == 0:
                self.core_source = True
            elif i == 1:
                self.jet_mixing = True
            elif i == 2:
                self.airframe_source = True
            elif i == 3:
                self.fan_inlet_source = True        
            elif i == 4:
                self.fan_discharge_source = True

            # Determine row and column in plot
            if np.remainder(i,3) == 0:
                irow = irow + 1
                icol = 0
            else:
                icol = icol + 1
            titles = ['Core', 'Jet mixing', 'Airframe', 'Fan inlet', 'Fan discharge']
            ax[irow,icol].set_title(titles[i], pad=-60)

            # Run noise source distribution
            pyna.initialize(self)
            if self.verification:
                self.data.load_source_verification_data(components=components)

            pyna.compute_noise_source_distribution(self, timestep=timestep)

            if metric == 'spl':
                # Plot frequencies 
                ci = -1
                k_plot = [3, 7, 10, 13, 17, 20, 23]
                theta = np.linspace(0, 180, 19)
                for k in np.arange(np.size(k_plot)):
                    ci = ci + 1

                    ax[irow,icol].set_thetamin(0)
                    ax[irow,icol].set_thetamax(180)

                    colors = plt.cm.magma(np.linspace(0,0.8,7))
                    if irow == 0 and icol==0:
                        if self.verification:
                            data_val = self.data.verification_source_supp[comp][26 * timestep : 26 * (timestep + 1) - 1, :]
                            data_val = data_val[1:,1:]
                            ax[irow,icol].plot( np.pi/180*theta[1:-1], data_val[k_plot[k],:], 'o', color=colors[ci])
                        ax[irow,icol].plot( np.pi/180*theta, self.noise_timeseries.get_val('noise.spl')[0, :, k_plot[k]], color=colors[ci], label='$f = $'+str(np.round(self.data.f[k_plot[k]]/1000.,1))+' $kHz$')
                    else:
                        if self.verification:
                            data_val = self.data.verification_source_supp[comp][26 * timestep : 26 * (timestep + 1) - 1, :]
                            data_val = data_val[1:,1:]
                            ax[irow,icol].plot( np.pi/180*theta[1:-1], data_val[k_plot[k],:], 'o', color=colors[ci])        
                        ax[irow,icol].plot( np.pi/180*theta, self.noise_timeseries.get_val('noise.spl')[0, :, k_plot[k]], color=colors[ci])

                ax[irow,icol].set_ylim([60, 150])
                ax[irow,icol].set_xticks(np.pi/180*np.array([0,45,90,135,180]))
                ax[irow,icol].set_yticks([60,90,120,150])
                ax[irow,icol].set_xlabel('SPL [dB]')
                ax[irow,icol].xaxis.set_label_coords(0.93, 0.15)

            elif metric == 'oaspl':
                if self.verification:
                    data_val = self.noise.data.verification_source_supp[comp][26 * timestep : 26 * (timestep + 1) - 1, :]
                    ax[irow,icol].plot(theta[1:-1], data_val[0, 1:], 'o')
                ax[irow,icol].plot( theta, self.noise_timeseries.get_val('noise.oaspl')[0, :])

                ax[irow,icol].set_xlabel(r'$\theta$ [deg]')
                ax[irow,icol].set_ylabel('OASPL [dB]')

        # Set legend
        ax[1,2].plot([0],[1], 'k-', label='pyNA')
        if self.settings.validation:
            ax[1,2].plot([0],[1], 'o', color='white', label='NASA STCA (Berton et al.)')
        if metric == 'spl':
            ax[0,0].legend(loc='lower left', bbox_to_anchor=(2.5, -0.35), ncol=2, borderaxespad=0, frameon=False)
        ax[1,2].legend(loc='lower left', bbox_to_anchor=(0.035, 0.3), ncol=1, borderaxespad=0, frameon=False)

        # Turn off the unused frames
        ax[1,2].axis('off')

        plt.show()

        return None

    def load_results(self, file_name, case_name='final'):
        """
        Load model .sql results file.

        :param file_name: Name of the .sql results file to load.
        :type file_name: str
        :param case_name: Name of the case to load in the results-file.
        :type case_name: str

        :return: results
        """

        # Create case reader
        cr = om.CaseReader(self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + file_name)
        results = cr.get_case(case_name)

        return results

    def save_time_series(self, problem:om.Problem, airframe: Airframe, path_save_name: str, engine_save_name: str) -> None:
        """
        Save a time_series solution of the trajectory calculations.

        :param problem: Openmdao probolem to save.
        :type problem: om.Problem
        :param airframe: airframe parameters
        :type airframe: airframe
        :param path_save_name: name of the time_series path save file
        :type path_save_name: str
        :param engine_save_name: name of the time_series engine save file
        :type engine_save_name: str

        :return: None
        """

        # Trajectory file
        t_source = problem.get_val('trajectory.t_s')
        t_intp = np.arange(t_source[0], t_source[-1], step=1)

        path = pd.DataFrame()
        path['t_source [s]'] = t_intp
        path['X [m]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.x'))
        path['Y [m]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.y'))
        path['Z [m]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.z'))
        path['V [m/s]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.v'))
        path['M_0 [-]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.M_0'))
        path['F_n [N]'] = np.interp(t_intp, t_source, problem.get_val('engine.F_n'))
        path['TS [-]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.TS'))
        path['c_0 [m/s]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.c_0'))
        path['T_0 [K]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.T_0'))
        path['p_0 [Pa]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.p_0'))
        path['rho_0 [kg/m3]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.rho_0'))
        path['mu_0 [kg/ms]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.mu_0'))
        path['I_0 [kg/m2s]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.I_0'))
        path['alpha [deg]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.alpha'))
        path['gamma [deg]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.gamma'))
        path['Airframe LG [-]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.I_landing_gear'))
        path['Airframe delta_f [deg]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.theta_flaps'))

        # Engine file
        engine = pd.DataFrame()
        engine['t_source [s]'] = t_intp
        engine['Ne [-]'] = airframe.n_eng * np.ones(np.size(t_intp))
        
        engine['Core mdot [kg/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.mdoti_c'))
        engine['Core Pt [Pa]'] = np.interp(t_intp, t_source, problem.get_val('engine.Pti_c'))
        engine['Core Tti [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Tti_c'))
        engine['Core Ttj [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Ttj_c'))
        engine['Core DT_t [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.DTt_des_c'))
            
        engine['LPT rho_e [kg/m3]'] = np.zeros(np.size(t_intp))
        engine['LPT c_e [m/s]'] = np.zeros(np.size(t_intp))
        engine['HPT rho_i [kg/m3]'] = np.zeros(np.size(t_intp))
        engine['HPT c_i [m/s]'] = np.zeros(np.size(t_intp))
        
        if self.jet_mixing_source == True and self.jet_shock_source == False:
            engine['Jet A [m2]'] = np.interp(t_intp, t_source, problem.get_val('engine.A_j'))
            engine['Jet rho [kg/m3]'] = np.interp(t_intp, t_source, problem.get_val('engine.rho_j'))
            engine['Jet Tt [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Tt_j'))
            engine['Jet V [m/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.V_j'))
        elif self.jet_mixing_source == False and self.jet_shock_source == True:
            engine['Jet A [m2]'] = np.interp(t_intp, t_source, problem.get_val('engine.A_j'))
            engine['Jet Tt [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Tt_j'))
            engine['Jet V [m/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.V_j'))
            engine['Jet M [-]'] = np.interp(t_intp, t_source, problem.get_val('engine.M_j'))
        elif self.jet_mixing_source == True and self.jet_shock_source == True:
            engine['Jet A [m2]'] = np.interp(t_intp, t_source, problem.get_val('engine.A_j'))
            engine['Jet rho [kg/m3]'] = np.interp(t_intp, t_source, problem.get_val('engine.rho_j'))
            engine['Jet Tt [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Tt_j'))
            engine['Jet V [m/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.V_j'))
            engine['Jet M [-]'] = np.interp(t_intp, t_source, problem.get_val('engine.M_j'))
        
        engine['Jet delta [deg]'] = np.zeros(np.size(t_intp))

        engine['Airframe LG [-]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.I_landing_gear'))
        engine['Airframe delta_f [deg]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.theta_flaps'))

        engine['Fan mdot in [kg/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.mdot_f'))
        engine['Fan A [m2]'] = np.interp(t_intp, t_source, problem.get_val('engine.A_f'))
        engine['Fan d [m]'] = np.interp(t_intp, t_source, problem.get_val('engine.d_f'))
        engine['Fan N [rpm]'] = np.interp(t_intp, t_source, problem.get_val('engine.N_f'))
        engine['Fan delta T [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.DTt_f'))
        engine['Fan B [-]'] = airframe.B_fan * np.ones(np.size(t_intp))
        engine['Fan V [-]'] = airframe.V_fan * np.ones(np.size(t_intp))
        engine['Fan RSS [%]'] = airframe.RSS_fan * np.ones(np.size(t_intp))
        engine['Fan IGV [-]'] = np.zeros(np.size(t_intp))
        engine['Fan ID [-]'] = np.ones(np.size(t_intp))

        # Save data frames
        path.to_csv(self.pyna_directory + '/cases/' + self.case_name + '/trajectory/' + self.output_directory_name + '/' + path_save_name)
        engine.to_csv(self.pyna_directory + '/cases/' + self.case_name + '/engine/' + self.output_directory_name + '/' + engine_save_name)

        return None

    @staticmethod
    def plot_optimizer_convergence_data(file_name: str) -> None:
        """
        Plot the convergence data of the optimization across iterates.

        :param file_name: name of the IPOPT output file
        :type file_name: str

        :return: None
        """

        # Read the IPOPT output file line by line
        myfile = open(file_name, 'rt')
        data = 'iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n'

        count = 0
        while True:
            # Get next line from file
            line = myfile.readline()

            # if line is empty: end of file is reached
            if not line:
                break

            # Look for iteration number
            if str(count) in line[:4]:
                count = count + 1
                # Remove r from the iteration line
                for tag in ['f', 'F', 'h', 'H', 'k', 'K', 'n', 'N', 'R', 'w', 's', 't', 'T', 'r']:
                    if tag in line:
                        line = line.replace(tag, '')

                # ADd line to data file
                data = data + line

        # Close the file
        myfile.close()

        # Write the file in csv format and convert to pandas data frame
        data = StringIO(data)
        data = pd.read_csv(data, delim_whitespace=True)

        # Plot
        fig, ax = plt.subplots(2, 3, figsize=(20, 8))
        plt.style.use('../utils/' + 'plot.mplstyle')

        ax[0, 0].plot(data['iter'].values, data['objective'].values)
        ax[0, 0].set_xlabel('Iterations')
        ax[0, 0].set_ylabel('Objective')
        ax[0, 0].tick_params(axis='both')
        ax[0, 0].grid(True)

        ax[0, 1].semilogy(data['iter'].values, data['inf_pr'], label='$inf_{pr}$')
        ax[0, 1].semilogy(data['iter'].values, data['inf_du'], label='$inf_{du}$')
        ax[0, 1].set_xlabel('Iterations')
        ax[0, 1].set_ylabel('Infeasibility')
        ax[0, 1].tick_params(axis='both')
        ax[0, 1].grid(True)
        ax[0, 1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)

        ax[0, 2].plot(data['iter'].values, data['lg(mu)'])
        ax[0, 2].set_xlabel('Iterations')
        ax[0, 2].set_ylabel('Barrier parameter')
        ax[0, 2].tick_params(axis='both')
        ax[0, 2].grid(True)
        # ax[0, 2].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)

        ax[1, 0].semilogy(data['iter'].values, data['||d||'].values)
        ax[1, 0].set_xlabel('Iterations')
        ax[1, 0].set_ylabel('||d||')
        ax[1, 0].tick_params(axis='both')
        ax[1, 0].grid(True)
        ax[1, 0].set_ylim(1e-10, 1e3)

        ax[1, 1].plot(data['iter'].values, data['alpha_pr'].values, label=r'$\alpha_{pr}$')
        ax[1, 1].plot(data['iter'].values, data['alpha_du'].values, label=r'$\alpha_{du}$')
        ax[1, 1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
        ax[1, 1].set_xlabel('Iterations')
        ax[1, 1].set_ylabel('Stepsize')
        ax[1, 1].tick_params(axis='both')
        ax[1, 1].grid(True)
        
        plt.subplots_adjust(hspace=0.45, wspace=0.3)

        fig.delaxes(ax[1, 2])

        return data

    @staticmethod
    def get_icao_annex_16_noise_limits(mtow, chapter: str, n_eng: int) -> dict:

        """
        
        :param mtow: maximum take-off weight [1000 kg]
        :type mtow:  np.ndarray
        :param chapter: ICAO Annex 16 Volume I Chapter
        :type chapter: str
        :param n_eng: number of installed engines [-]
        :type n_eng: int

        """

        limits = dict()
        limits['lateral'] = np.zeros(np.size(mtow))
        limits['flyover'] = np.zeros(np.size(mtow))
        limits['approach'] = np.zeros(np.size(mtow))

        # ICAO Chapter 3 limits
        if chapter == '3':
            limits['lateral'][mtow <= 35] = 94*np.ones(np.size(mtow))[mtow <= 35]
            limits['lateral'][(35<mtow)*(mtow<=400)]= (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 400)]
            limits['lateral'][400<mtow]=103*np.ones(np.size(mtow))[400 < mtow]

            limits['approach'][mtow<=35] = 98*np.ones(np.size(mtow))[mtow <= 35]
            limits['approach'][(35<mtow)*(mtow<=280)]= (86.03 + 7.75*np.log10(mtow))[(35 < mtow)*(mtow<=280)]
            limits['approach'][280<mtow]=105*np.ones(np.size(mtow))[280<mtow]

            if n_eng == 2:
                limits['flyover'][mtow <= 48.1] = 89*np.ones(np.size(mtow))[mtow <= 48.1] 
                limits['flyover'][(48.1 < mtow)*(mtow<= 385)] = (66.65 + 13.29*np.log10(mtow))[(48.1 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 101*np.ones(np.size(mtow))[385 < mtow] 
            elif n_eng == 3:
                limits['flyover'][mtow <= 28.6] = 89*np.ones(np.size(mtow))[mtow <= 28.6] 
                limits['flyover'][(28.6 < mtow)*(mtow<= 385)] = (69.65 + 13.29*np.log10(mtow))[(28.6 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 104*np.ones(np.size(mtow))[385 < mtow] 
            elif n_eng == 4:
                limits['flyover'][mtow <= 20.2] = 89*np.ones(np.size(mtow))[mtow <= 20.2] 
                limits['flyover'][(20.2 < mtow)*(mtow<= 385)] = (71.65 + 13.29*np.log10(mtow))[(20.2 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 106*np.ones(np.size(mtow))[385 < mtow] 
            else:
                raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

        # ICAO Chapter 14 limits
        elif chapter == '14':
            limits['lateral'][mtow <= 2] = 88.6*np.ones(np.size(mtow))[mtow <= 2]
            limits['lateral'][(2 < mtow)*(mtow<= 8.618)]= (86.03754 + 8.512295*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
            limits['lateral'][(8.618 < mtow)*(mtow<= 35)]= 94*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 35)]
            limits['lateral'][(35 < mtow)*(mtow<= 400)]= (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 400)]
            limits['lateral'][400 < mtow] = 103*np.ones(np.size(mtow))[400 < mtow]

            limits['approach'][mtow <= 2] = 93.1*np.ones(np.size(mtow))[mtow <= 2]
            limits['approach'][(2 < mtow)*(mtow<= 8.618)]= (90.77481 + 7.72412*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
            limits['approach'][(8.618 < mtow)*(mtow<= 35)]= 98*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 35)]
            limits['approach'][(35<mtow)*(mtow<= 280)]= (86.03167 + 7.75117*np.log10(mtow))[(35 < mtow)*(mtow<= 280)]
            limits['approach'][280<mtow] = 105*np.ones(np.size(mtow))[280<mtow]

            if n_eng == 2:
                limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
                limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
                limits['flyover'][(8.618 < mtow)*(mtow<=48.125)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<=48.125)]
                limits['flyover'][(48.125 < mtow)*(mtow<= 385)] = (66.65 + 13.29*np.log10(mtow))[(48.125 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 101*np.ones(np.size(mtow))[385 < mtow] 
            elif n_eng == 3:
                limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
                limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
                limits['flyover'][(8.618 < mtow)*(mtow<= 28.615)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 28.615)]
                limits['flyover'][(28.615 < mtow)*(mtow<= 385)] = (69.65 + 13.29*np.log10(mtow))[(28.615 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 104*np.ones(np.size(mtow))[385 < mtow] 
            elif n_eng == 4:
                limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
                limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
                limits['flyover'][(8.618 < mtow)*(mtow<= 20.234)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 20.234)]
                limits['flyover'][(20.234 < mtow)*(mtow<= 385)] = (71.65 + 13.29*np.log10(mtow))[(20.234 < mtow)*(mtow<= 385)]
                limits['flyover'][385 < mtow] = 106*np.ones(np.size(mtow))[385 < mtow] 
            else:
                raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

            # Apply the noise margins
            limits['cumulative'] = (limits['lateral']+limits['flyover']+limits['approach']) - 17
            limits['lateral'] = limits['lateral'] - 1
            limits['flyover'] = limits['flyover'] - 1
            limits['approach'] = limits['approach'] - 1

        # FAA NPRM noise limits
        elif chapter == 'NPRM':

            limits['lateral'][mtow <= 35] = 94*np.ones(np.size(mtow))[mtow <= 35]
            limits['lateral'][(35 < mtow)*(mtow<= 68.039)] = (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 68.039)]
            limits['lateral'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]

            limits['approach'][mtow <= 35] = 98*np.ones(np.size(mtow))[mtow <= 35]
            limits['approach'][(35 < mtow)*(mtow<= 68.039)] = (86.03167 + 7.75117*np.log10(mtow))[(35 < mtow)*(mtow<= 68.039)]
            limits['approach'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]

            if n_eng == 2:
                limits['flyover'][mtow <= 48.125] = 89*np.ones(np.size(mtow))[mtow <= 48.125]
                limits['flyover'][(48.125 < mtow)*(mtow<= 68.039)] = (66.65 + 13.29*np.log10(mtow))[(48.125 < mtow)*(mtow<=68.039)]
                limits['flyover'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]
            elif n_eng == 3:
                limits['flyover'][mtow <= 28.615] = 89*np.ones(np.size(mtow))[mtow <= 28.615]
                limits['flyover'][(28.615 < mtow)*(mtow<= 68.039)] = (69.65 + 13.29*np.log10(mtow))[(28.615 < mtow)*(mtow<= 68.039)]
                limits['flyover'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]
            else:
                raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

        else:
            raise ValueError("ICAO Chapter " + chapter + "noise limits are not available. Specify '3', '14', or 'NPRM'.")

        return limits



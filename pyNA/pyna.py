import pdb
import copy
import os
import numpy as np
import pandas as pd
from io import StringIO
import openmdao.api as om
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.engine import Engine
from pyNA.src.trajectory import Trajectory
from pyNA.src.noise import Noise

class pyna:
    """
    The pyna module contains the methods to assess the noise footprint of an aircraft (i.e airframe and engine) flying
    along a predefined trajectory and calculates the sensitivities of the noise footprint with respect to engine
    variables.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize pyna class.

        :param settings: pyna settings
        :type settings: Settings

        :return: None

        """

        # Set pyNA settings class and check
        self.settings = copy.copy(settings)
        self.settings.check()

        # Settings
        self.settings.pyNA_directory = os.path.dirname(__file__)
        self.settings.language = os.environ['pyna_language']

        # Initialize trajectory
        self.trajectory = Trajectory(n_order=self.settings.n_order)
        
        # Initialize noise
        self.noise = Noise(settings=self.settings)

        # Initialize aircraft configuration
        self.ac = Aircraft(name=self.settings.ac_name, version=settings.ac_version, settings=self.settings)
        
        # Initialize engine configuration
        self.engine = Engine()

        # Initialize openmdao problems
        self.problem_init = om.Problem()
        self.problem = om.Problem()

    # --- Noise-only methods --------------------------------------------------------------------------------
    @staticmethod
    def load_settings(case_name: str) -> Settings:
        """
        Load default pyna settings.

        :param case_name: name of the case to load
        :type case_name: str

        :return: pyna settings
        :rtype: Settings
        """

        return Settings(case_name)

    def compute_noise_time_series(self) -> None:
        """
        Compute noise for a predefined trajectory from .csv files.

        :return: None
        """

        # Load aircraft and engine timeseries
        self.engine.load_time_series(settings=self.settings, engine_file_name=self.settings.engine_file_name)

        # Load trajectory data
        self.trajectory.load_time_series(settings=self.settings)

        # Create noise
        self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/coloring_files/')
        self.noise.setup_time_series(problem=self.problem, settings=self.settings, ac=self.ac, n_t=self.trajectory.n_t, mode='trajectory', objective='time_series')
        self.noise.compute_time_series(problem=self.problem, settings=self.settings, path=self.trajectory.path, engine=self.engine.time_series, mode='trajectory')

        return None

    def compute_noise_contours(self, x_lst: np.ndarray, y_lst: np.ndarray) -> None:
        """
        Compute noise contours for a predefined trajectory.

        :param x_lst: List of x-location of the microphones.
        :type x_lst: list
        :param y_lst: List of y-location of the microphones.
        :type y_lst: list

        :return: None
        """

        # Load aircraft and engine time series
        self.engine.load_time_series(settings=self.settings, engine_file_name=self.settings.engine_file_name)

        # Load trajectory data
        self.trajectory.load_time_series(settings=self.settings)

        # Get list of observers
        self.settings.x_observer_array = np.zeros((np.size(x_lst)*np.size(y_lst), 3))
        cntr = -1
        for i, y in enumerate(y_lst):
            for j, x in enumerate(x_lst):
                cntr = cntr + 1
                self.settings.x_observer_array[cntr, 0] = x
                self.settings.x_observer_array[cntr, 1] = y
                self.settings.x_observer_array[cntr, 2] = 4*0.3048

        self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/coloring_files/')
        self.noise.setup_time_series(problem=self.problem, settings=self.settings, ac=self.ac, n_t=self.trajectory.n_t, mode='trajectory', objective='contours')
        self.noise.compute_time_series(problem=self.problem, settings=self.settings, path=self.trajectory.path, engine=self.engine.time_series, mode='trajectory')

        # Extract the contour
        self.noise.contour['x_lst'] = x_lst
        self.noise.contour['y_lst'] = np.hstack((-np.flip(y_lst[1:]), y_lst))
        self.noise.contour['X'], self.noise.contour['Y'] = np.meshgrid(self.noise.contour['x_lst'], self.noise.contour['y_lst'])
        
        # Initialize contour solution matrix
        epnl_contour = np.zeros((np.size(y_lst), np.size(x_lst)))
        cntr = -1
        for i, y in enumerate(y_lst):
            for j, x in enumerate(x_lst):
                cntr = cntr + 1
                epnl_contour[i,j] = self.problem.get_val('noise.'+self.settings.levels_int_metric)[cntr]

        # Flip epnl solution matrix for negative y-values
        self.noise.contour[self.settings.levels_int_metric] = np.vstack((np.flipud(epnl_contour[1:, :]), epnl_contour))

        return None

    def compute_noise_epnl_table(self) -> pd.DataFrame:

        """
        Compute table of epnl for individual noise sources and observers.

        :return: None
        """

        # Set levels_int_metric to epnl
        self.settings.levels_int_metric = 'epnl'

        # Initialize epnl
        epnl_table = np.zeros((6, len(self.settings.observer_lst)+1))

        # Iterate over component list
        components = ['fan_inlet', 'fan_discharge', 'core', 'jet_mixing', 'airframe', 'all_sources']
        for i, comp in enumerate(components):

            # Reset component flags
            self.settings.fan_inlet = False
            self.settings.fan_discharge = False
            self.settings.core = False
            self.settings.jet_mixing = False
            self.settings.jet_shock = False
            self.settings.airframe = False
            self.settings.all_sources = False

            # Enable component flag
            if comp == 'fan_inlet':
                self.settings.fan_inlet = True
            elif comp == 'fan_discharge':
                self.settings.fan_discharge = True
            elif comp == 'core':
                self.settings.core = True
            elif comp == 'jet_mixing':
                self.settings.jet_mixing = True
            elif comp == 'jet_shock':
                self.settings.jet_shock = True
            elif comp == 'airframe':
                self.settings.airframe = True
            elif comp == 'all_sources':
                self.settings.fan_inlet = True
                self.settings.fan_discharge = True
                self.settings.core = True
                self.settings.jet_mixing = True
                self.settings.jet_shock = True
                self.settings.airframe = True

            # Run pyNA
            pyna.compute_noise_time_series(self)

            # Save solutions
            for j in np.arange(len(self.settings.observer_lst)):
                epnl_table[i, j] = np.round(self.problem.get_val('noise.epnl')[j], 1)
            epnl_table[i, j+1] = np.round(np.sum(self.problem.get_val('noise.epnl')), 1)

        # Create data frame for solutions
        self.noise.epnl_table = pd.DataFrame(epnl_table)
        observer_lst = list(self.settings.observer_lst) + ['take-off']
        self.noise.epnl_table.columns = observer_lst
        self.noise.epnl_table.index = components

        return self.noise.epnl_table

    def compute_noise_source_distribution(self, time_step: np.int64) -> None:
        """
        Compute noise source spectral and directional distribution.

        :param time_step: Time step in predefined trajectory at which to compute the noise source distribution.
        :type time_step: np.int64

        :return: None
        """

        # Only implemented for python
        if self.settings.language == 'julia':
            raise ValueError('This method has not yet been implemented in Julia. Set os.environ["language"]="python" and run again.')

        # Load aircraft and engine time series
        self.engine.load_operating_point(settings=self.settings, time_step=time_step, engine_file_name=self.settings.engine_file_name)

        # Load trajectory data
        self.trajectory.load_operating_point(settings=self.settings, time_step=time_step)

        # Single observer setting; disable shielding
        self.settings.shielding = False

        # Create noise
        self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/coloring_files/')
        self.noise.setup_time_series(problem=self.problem, settings=self.settings, ac=self.ac, n_t=self.trajectory.n_t, mode='distribution', objective=None)
        self.noise.compute_time_series(problem=self.problem, settings=self.settings, path=self.trajectory.path, engine=self.engine.time_series, mode='distribution')

        return None

    # --- Trajectory-only methods ---------------------------------------------------------------------------
    def compute_trajectory(self, trajectory_mode='cutback', objective='t_end') -> bool:
        """
        Compute aircraft take-off trajectory.

        :return: converged
        :rtype: bool 

        """

        # Load aircraft and engine deck
        self.ac.load_aerodynamics(settings=self.settings)
        self.engine.load_deck(settings=self.settings)

        # Create trajectory
        self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/output/' + self.settings.output_directory_name + '/' + 'coloring_files/')
        self.trajectory.setup(problem=self.problem, settings=self.settings, ac=self.ac, engine=self.engine, trajectory_mode=trajectory_mode, objective=objective)
        self.trajectory.compute(problem=self.problem, settings=self.settings, ac=self.ac, run_driver=True, init_trajectory=None, trajectory_mode=trajectory_mode, objective=objective)

        # Check convergence
        converged = self.trajectory.check_convergence(settings=self.settings, filename='IPOPT_trajectory_convergence.out')

        return converged

    # --- Noise-trajectory methods --------------------------------------------------------------------------
    def compute_trajectory_noise(self, objective='t_end', trajectory_mode='cutback', init_traj_name=None) -> None:
        """
        Compute aircraft take-off trajectory and noise.
        
        :param objective: optimization objective
        :type objective: str
        :param init_traj_name: Name of initialization trajectory (in output folder of case).
        :type init_traj_name: str

        :return: None
        """

        # Load aircraft and engine deck
        self.ac.load_aerodynamics(settings=self.settings)
        self.engine.load_deck(settings=self.settings)

        # Create initialization trajectory
        if not init_traj_name:
            # Compute trajectory
            self.problem_init = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/output/' + self.settings.output_directory_name + '/coloring_files/')
            self.trajectory.setup(problem=self.problem_init, settings=self.settings, ac=self.ac, engine=self.engine, trajectory_mode=trajectory_mode, objective=objective)
            self.trajectory.compute(problem=self.problem_init, settings=self.settings, ac=self.ac, run_driver=True, init_trajectory=None, trajectory_mode=trajectory_mode, objective=objective)

            # Check convergence
            converged = self.trajectory.check_convergence(settings=self.settings, filename='IPOPT_trajectory_convergence.out')
        else:
            # Load trajectory
            self.problem_init = pyna.load_results(self, init_traj_name, 'final')

            # Convergence
            converged = True

        # Setup trajectory for noise computations
        if converged:
            self.trajectory.n_t = np.size(self.problem_init.get_val('trajectory.t_s'))
            self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/coloring_files/')
            self.trajectory.setup(problem=self.problem, settings=self.settings, ac=self.ac, engine=self.engine, trajectory_mode=trajectory_mode, objective=None)
            self.noise.setup_trajectory_noise(problem=self.problem, settings=self.settings, ac=self.ac, n_t=self.trajectory.n_t, objective='noise')

            # Run the noise calculations
            self.trajectory.compute(problem=self.problem, settings=self.settings, ac=self.ac, run_driver=False, init_trajectory=self.problem_init, trajectory_mode=trajectory_mode, objective=None)
        else:
            print('Trajectory did not converge.')

        return None

    def optimize_trajectory_noise(self, n_sideline=1, init_traj_name=None) -> None:
        """
        Optimize aircraft take-off trajectory for minimum noise signature.

        :param n_sideline: Number of sideline microphones to use in the control optimization
        :type n_sideline: int
        :param init_traj_name: Name of initialization trajectory (in output folder of case).
        :type init_traj_name: str

        :return: None
        """

        # Load aircraft and engine deck
        self.ac.load_aerodynamics(settings=self.settings)
        self.engine.load_deck(settings=self.settings)

        # Create initialization trajectory
        if not init_traj_name:
            self.problem_init = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/output/' + self.settings.output_directory_name + '/coloring_files/')
            self.trajectory.setup(problem=self.problem_init, settings=self.settings, ac=self.ac, engine=self.engine, trajectory_mode='flyover', objective='t_end')
            self.trajectory.compute(problem=self.problem_init, settings=self.settings, ac=self.ac, run_driver=True, init_trajectory=None, trajectory_mode='flyover', objective='t_end') 
        else:
            self.problem_init = pyna.load_results(self, init_traj_name, 'final')

        # Get list of observers
        self.settings.x_observer_array = np.zeros((n_sideline+1, 3))
        self.settings.x_observer_array[:-1, 0] = np.linspace(1000, 5500, n_sideline)
        self.settings.x_observer_array[:-1, 1] = 450.
        self.settings.x_observer_array[:-1, 2] = 4 * 0.3048
        self.settings.x_observer_array[-1, 0] = 6500.
        self.settings.x_observer_array[-1, 1] = 0.
        self.settings.x_observer_array[-1, 2] = 4 * 0.3048
        
        # Set k_rot
        self.ac.k_rot = self.problem_init.get_val('phases.groundroll.parameters:k_rot')

        # Setup trajectory for noise computations
        self.trajectory.n_t = np.size(self.problem_init.get_val('trajectory.t_s'))
        self.problem = om.Problem(coloring_dir=self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/coloring_files/')
        self.trajectory.setup(problem=self.problem, settings=self.settings, ac=self.ac, engine=self.engine, trajectory_mode='flyover', objective='noise')
        self.noise.setup_trajectory_noise(problem=self.problem, settings=self.settings, ac=self.ac, n_t=self.trajectory.n_t, objective='noise')

        # Run the trajectory
        self.trajectory.compute(problem=self.problem, settings=self.settings, ac=self.ac, run_driver=True, init_trajectory=self.problem_init, trajectory_mode='flyover', objective='noise')

        return None
    
    # --- Post-processing of the results --------------------------------------------------------------------
    @staticmethod
    def save_time_series(problem: om.Problem, settings: Settings, ac: Dict[str, Any], path_save_name: str, engine_save_name: str) -> None:
        """
        Save a time_series solution of the trajectory calculations.

        :param problem: Openmdao probolem to save.
        :type problem: om.Problem
        :param settings: pyna settings
        :type settings: Settings
        :param ac: aircraft parameters
        :type ac: Aircraft
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
        path['F_n [N]'] = np.interp(t_intp, t_source, problem.get_val('trajectory.F_n'))
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
        engine['Ne [-]'] = ac.n_eng * np.ones(np.size(t_intp))
        
        engine['Core mdot [kg/s]'] = np.interp(t_intp, t_source, problem.get_val('engine.mdoti_c'))
        engine['Core Pt [Pa]'] = np.interp(t_intp, t_source, problem.get_val('engine.Pti_c'))
        engine['Core Tti [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Tti_c'))
        engine['Core Ttj [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.Ttj_c'))
        engine['Core DT_t [K]'] = np.interp(t_intp, t_source, problem.get_val('engine.DTt_des_c'))
        engine['LPT rho_e [kg/m3]'] = np.zeros(np.size(t_intp))
        engine['LPT c_e [m/s]'] = np.zeros(np.size(t_intp))
        engine['HPT rho_i [kg/m3]'] = np.zeros(np.size(t_intp))
        engine['HPT c_i [m/s]'] = np.zeros(np.size(t_intp))
        
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
        engine['Fan B [-]'] = ac.B_fan * np.ones(np.size(t_intp))
        engine['Fan V [-]'] = ac.V_fan * np.ones(np.size(t_intp))
        engine['Fan RSS [%]'] = ac.RSS_fan * np.ones(np.size(t_intp))
        engine['Fan IGV [-]'] = np.zeros(np.size(t_intp))
        engine['Fan ID [-]'] = np.ones(np.size(t_intp))

        # Save data frames
        path.to_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/trajectory/' + settings.output_directory_name + '/' + path_save_name)
        engine.to_csv(settings.pyNA_directory + '/cases/' + settings.case_name + '/engine/' + settings.output_directory_name + '/' + engine_save_name)

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
        cr = om.CaseReader(self.settings.pyNA_directory + '/cases/' + self.settings.case_name + '/output/' + self.settings.output_directory_name + '/' + file_name)
        results = cr.get_case(case_name)

        return results

    # --- Plotting functions --------------------------------------------------------------------------------
    def plot_trajectory(self, problem, *problem_verify):
        # Check if problem_verify is empty
        if problem_verify:
            verification = True
            problem_verify = problem_verify[0]
        else:
            verification = False
        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use(self.settings.pyNA_directory + '/utils/' + 'plot.mplstyle')

        ax[0,0].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.z'), '-', label='Take-off trajectory module')
        if verification:
            ax[0,0].plot(problem_verify['X [m]'], problem_verify['Z [m]'], '--', label='NASA STCA (Berton)')
        ax[0,0].set_xlabel('t [s]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)

        ax[0,1].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.v'), '-')
        if verification:
            ax[0,1].plot(problem_verify['t_source [s]'], problem_verify['V [m/s]'], '--', label='NASA STCA (Berton)')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')

        ax[0,2].plot(problem.get_val('trajectory.t_s'), (np.arctan(0.04)*180/np.pi)*np.ones(np.size(problem.get_val('trajectory.t_s'))), 'k--')
        ax[0,2].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.gamma'), '-')
        if verification:
            ax[0,2].plot(problem_verify['t_source [s]'], problem_verify['gamma [deg]'], '--')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')

        ax[1,0].plot(problem.get_val('trajectory.t_s'), 1 / 1000. * problem.get_val('trajectory.F_n'), '-')
        if verification:
            ax[1,0].plot(problem_verify['t_source [s]'], 1 / 1000. * problem_verify['F_n [N]'], '--')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')

        ax[1,1].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.alpha'), '-')
        if verification:
            ax[1,1].plot(problem_verify['t_source [s]'], problem_verify['alpha [deg]'], '--')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$\alpha$ [deg]')

        colors=['tab:blue', 'tab:orange']
        ax[1,2].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.theta_flaps'), '-', label=r'$\theta_{flaps}$', color=colors[0])
        ax[1,2].plot(problem.get_val('trajectory.t_s'), problem.get_val('trajectory.theta_slats')/(-1), '-', label=r'$\theta_{slats}$', color=colors[1])
        if verification:
            ax[1,2].plot(problem_verify['t_source [s]'], problem_verify['Airframe delta_f [deg]'], '--', color=colors[0])
            ax[1,2].plot(problem_verify['t_source [s]'], problem_verify['Airframe delta_s [deg]']/(-1), '--', color=colors[1])
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\theta$ [deg]')
        ax[1,2].set_ylim([-2, 28])
        ax[1,2].legend(ncol=1, borderaxespad=0, frameon=False)
 
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()

        return None

    def plot_noise_time_series(self, metric: str) -> None:
        """
        Plot the noise metric along the trajectory.

        :param metric: noise metric to plot. Specify 'pnlt' or 'oaspl'/
        :type metric: str

        :return: None
        """

        # Create figure
        fig, ax = plt.subplots(1, len(self.settings.observer_lst), figsize=(20, 4.3), dpi=100)
        ax_zoom = copy.copy(ax)
        plt.style.use(self.settings.pyNA_directory + '/utils/' + 'plot.mplstyle')

        colors = plt.cm.magma(np.linspace(0,0.8,2))

        # Iterate over observer locations
        for i, observer in enumerate(self.settings.observer_lst):

            # Time range of epnl domain of dependence
            time_epnl = self.problem.get_val('noise.t_o')[i,:][np.where(self.problem.get_val('noise.pnlt')[i,:] > max(self.problem.model.get_val('noise.pnlt')[i,:]) - 10.)]

            # Plot noise levels
            if metric == 'pnlt':
                # Plot noise levels
                if observer == 'lateral':
                    ax[i].plot(self.problem.model.get_val('noise.t_o')[i,:180], self.problem.model.get_val('noise.pnlt')[i,:180], linewidth=2.5, label='pyNA', color=colors[0])
                else:
                    ax[i].plot(self.problem.model.get_val('noise.t_o')[i,:], self.problem.model.get_val('noise.pnlt')[i,:], linewidth=2.5, label='pyNA', color=colors[0])
                ax[i].fill_between([time_epnl[0], time_epnl[-1]], [-5, -5], [105, 105], alpha=0.15,
                                label='EPNL domain of dependence', color=colors[0])
                if self.settings.validation:
                    self.noise.data.load_trajectory_verification_data(settings=self.settings)
                    ax[i].plot(self.noise.data.verification_trajectory[observer]['t observer [s]'],
                            self.noise.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, label='NASA STCA (Berton et al. [25])', color=colors[1])

                ax[i].grid(True)
                ax[i].set_xlabel('Time after brake release [s]')
                ax[i].tick_params(axis='both')
                ax[i].set_ylim([-5, 105])

                # Zoomed-in plots
                ax_zoom[i] = zoomed_inset_axes(ax[i], zoom=4, loc='lower right')
                ax_zoom[i].plot(self.problem.model.get_val('noise.t_o')[i,:180], self.problem.model.get_val('noise.pnlt')[i,:180], linewidth=2.5, color=colors[0])
                if self.settings.validation:
                    ax_zoom[i].plot(self.noise.data.verification_trajectory[observer]['t observer [s]'], self.noise.data.verification_trajectory[observer]['PNLT'], '--', linewidth=2.5, color=colors[1])
                ax_zoom[i].set_xticks([])
                ax_zoom[i].set_yticks([])
                ax_zoom[i].set_xlim([time_epnl[0], time_epnl[-1]])
                ax_zoom[i].set_ylim([np.max(self.problem.get_val('noise.pnlt')[i,:])-11, np.max(self.problem.get_val('noise.pnlt')[i,:])+1.5])
                mark_inset(ax[i], ax_zoom[i], loc1=1, loc2=3)                

            elif metric == 'oaspl':
                ax[i].plot(self.problem.model.get_val('noise.t_o')[i,:], self.problem.model.get_val('noise.oaspl')[i,:], linewidth=2.5, label='pyNA')
                ax[i].set_ylabel('$OASPL_{' + observer + '}$ [dB]')
                if self.settings.validation:
                    self.noise.data.load_trajectory_verification_data(settings=self.settings)
                    ax[i].plot(self.noise.data.verification_trajectory[observer]['t observer [s]'],
                               self.noise.data.verification_trajectory[observer]['OASPL'], '--', linewidth=2.5,
                               label='NASA STCA (Berton et al. [25])')

            ax[i].grid(True)
            ax[i].set_xlabel('Time after brake release [s]')
            ax[i].tick_params(axis='both')
            ax[i].set_ylim([-5, 105])

        ax[0].set_title('Lateral')
        ax[1].set_title('Flyover')
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
        plt.style.use(self.settings.pyNA_directory + '/utils/' + 'plot.mplstyle')

        CS = plt.contour(self.noise.contour['X'], self.noise.contour['Y'], self.noise.contour['epnl'], levels=[60, 65, 70, 75, 80, 85, 90, 100, 110])
        plt.plot(self.trajectory.path['X [m]'], self.trajectory.path['Y [m]'], 'k-', linewidth=4, label='Trajectory')

        plt.clabel(CS, [60, 65, 70, 75, 80, 85, 90, 100, 110], inline=True, fmt=fmt, fontsize=18)
        plt.xlim([np.min(self.noise.contour['X']), np.max(self.noise.contour['X'])])
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.legend()
        plt.show()

        return None

    def plot_noise_source_distribution(self, time_step:np.int64, metric:str, components:list) -> None:
        
        # Plot noise hemispheres 
        if metric == 'spl':
            fig, ax = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(10, 10), dpi=100)
            plt.subplots_adjust(wspace=-0.,hspace=-0.2)
        elif metric == 'oaspl':
            fig, ax = plt.subplots(2, 3, figsize=(20, 8))
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
        else:
            raise ValueError('Invalid metric specified. Specify: spl / oaspl.')
        plt.style.use(self.settings.pyNA_directory + '/utils/' + 'plot.mplstyle')

        # Initialize pyna
        if self.settings.validation:
            self.noise.data.load_source_verification_data(settings=self.settings, components=components)

        # Loop through different components
        irow = -1
        icol = -1
        for i, comp in enumerate(components):

            # Enable component in pyna settings
            self.settings.all_sources = False
            self.settings.fan_inlet = False
            self.settings.fan_discharge = False
            self.settings.core = False
            self.settings.jet_mixing = False
            self.settings.jet_shock = False
            self.settings.airframe = False
            if i == 0:
                self.settings.core = True
            elif i == 1:
                self.settings.jet_mixing = True
            elif i == 2:
                self.settings.airframe = True
            elif i == 3:
                self.settings.fan_inlet = True        
            elif i == 4:
                self.settings.fan_discharge = True

            # Determine row and column in plot
            if np.remainder(i,3) == 0:
                irow = irow + 1
                icol = 0
            else:
                icol = icol + 1
            titles = ['Core', 'Jet mixing', 'Airframe', 'Fan inlet', 'Fan discharge']
            ax[irow,icol].set_title(titles[i], pad=-60)

            # Run noise source distribution
            theta = np.linspace(0, 180, 19)
            self.compute_noise_source_distribution(time_step=time_step)

            if metric == 'spl':
                # Plot frequencies 
                ci = -1
                k_plot = [3, 7, 10, 13, 17, 20, 23]
                for k in np.arange(np.size(k_plot)):
                    ci = ci + 1

                    ax[irow,icol].set_thetamin(0)
                    ax[irow,icol].set_thetamax(180)

                    colors = plt.cm.magma(np.linspace(0,0.8,7))
                    if irow == 0 and icol==0:
                        if self.settings.validation:
                            data_val = self.noise.data.verification_source_supp[comp][26 * time_step : 26 * (time_step + 1) - 1, :]
                            data_val = data_val[1:,1:]
                            ax[irow,icol].plot( np.pi/180*theta[1:-1], data_val[k_plot[k],:], 'o', color=colors[ci])
                        ax[irow,icol].plot( np.pi/180*theta, self.problem.get_val('noise.spl')[0, :, k_plot[k]], color=colors[ci], label='$f = $'+str(np.round(self.noise.data.f[k_plot[k]]/1000.,1))+' $kHz$')
                    else:
                        if self.settings.validation:
                            data_val = self.noise.data.verification_source_supp[comp][26 * time_step : 26 * (time_step + 1) - 1, :]
                            data_val = data_val[1:,1:]
                            ax[irow,icol].plot( np.pi/180*theta[1:-1], data_val[k_plot[k],:], 'o', color=colors[ci])        
                        ax[irow,icol].plot( np.pi/180*theta, self.problem.get_val('noise.spl')[0, :, k_plot[k]], color=colors[ci])

                ax[irow,icol].set_ylim([60, 150])
                ax[irow,icol].set_xticks(np.pi/180*np.array([0,45,90,135,180]))
                ax[irow,icol].set_yticks([60,90,120,150])
                ax[irow,icol].set_xlabel('SPL [dB]')
                ax[irow,icol].xaxis.set_label_coords(0.93, 0.15)

            elif metric == 'oaspl':
                if self.settings.validation:
                    data_val = self.noise.data.verification_source_supp[comp][26 * time_step : 26 * (time_step + 1) - 1, :]
                    ax[irow,icol].plot(theta[1:-1], data_val[0, 1:], 'o')
                ax[irow,icol].plot( theta, self.problem.get_val('noise.oaspl')[0, :])

                ax[irow,icol].set_xlabel(r'$\theta$ [deg]')
                ax[irow,icol].set_ylabel('OASPL [dB]')

        # Set legend
        ax[1,2].plot([0],[1], 'k-', label='pyNA')
        if self.settings.validation:
            ax[1,2].plot([0],[1], 'o', color='white', label='NASA STCA (Berton)')
        if metric == 'spl':
            ax[0,0].legend(loc='lower left', bbox_to_anchor=(2.5, -0.35), ncol=2, borderaxespad=0, frameon=False)
        ax[1,2].legend(loc='lower left', bbox_to_anchor=(0.035, 0.3), ncol=1, borderaxespad=0, frameon=False)

        # Turn off the unused frames
        ax[1,2].axis('off')

        plt.show()

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
        plt.subplots_adjust(hspace=0.45, wspace=0.4)

        fig.delaxes(ax[1, 2])

        return data
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import openmdao.api as om
import os
import pdb
import copy
import dymos as dm
os.environ["OPENMDAO_REPORTS"] = 'none'
os.environ["PYNA_LANGUAGE"] = 'python'
import pyNA
from pyNA.src.problem import Problem
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory import Trajectory
from pyNA.src.time_history import TimeHistory
from pyNA.src.stca_data import StcaData
from pyNA.src.noise import Noise
if os.environ["PYNA_LANGUAGE"] == 'julia':
    from pyNA.src.noise_jl import NoiseJl


class pyna:

    """
    
    Parameters
    ----------

    Attributes
    ----------
    problem : Problem
        pyNA model 
    settings : dict
        pyNA settings
    aircraft : Aircraft
        _
    trajectory : Trajectory
        _
    noise : Noise
        _

    """


    def __init__(self, 
                case_name = 'nasa_stca_standard',
                aircraft_name = 'stca',
                engine_name = 'engine_berton',
                output_directory_name = '',
                output_file_name = 'trajectory_stca.sql',
                time_history_file_name = 'time_history_to.csv',
                engine_deck_file_name = 'engine_deck_stca.csv',
                atmosphere_mode = 'stratified',
                fan_BB_method = 'geae',
                fan_RS_method = 'allied_signal',
                fan_ge_flight_cleanup = 'takeoff',
                core_turbine_attenuation_method = 'ge',
                lateral_attenuation_engine_mounting = 'underwing',
                levels_int_metric = 'epnl',
                observer_lst = ('lateral', 'flyover'),
                trajectory_mode = 'time_history',
                thrust_lapse = True,
                ptcb = False,
                phld = False,
                pkrot = False,
                all_sources = True,
                fan_inlet_source = True,
                fan_discharge_source = True,
                core_source = True,
                jet_mixing_source = True,
                jet_shock_source = False,
                airframe_source = True,
                fan_igv = False,
                fan_id = False,
                fan_combination_tones = False,
                fan_liner_suppression = True,
                airframe_hsr_calibration = True,
                direct_propagation = True,
                absorption = True,
                ground_effects = True,
                lateral_attenuation = True,
                shielding = True,
                tones_under_800Hz = False,
                epnl_bandshare = False,
                core_jet_suppression = True,
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
                x_microphones = np.array([[12325.*0.3048, 450., 4*0.3048], [21325.*0.3048, 0., 4*0.3048]]),
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
        
        Parameters
        ----------
        case_name : str
            _
        aircraft_name : str
            _
        engine_name : str
            _
        output_directory_name : str
            _
        output_file_name : str
            _
        time_history_file_name : str
            _
        engine_deck_file_name : str
            _
        atmosphere_mode : str
            _
        fan_BB_method : str
            _
        fan_RS_method : str
            _
        fan_ge_flight_cleanup : str
            _
        core_turbine_attenuation_method : str
            _
        lateral_attenuation_engine_mounting : str
            _
        levels_int_metric : str
            _
        param observer_lst : list
            _
        trajectory_mode : string
            _
        thrust_lapse : bool
            _
        ptcb : bool
            _
        phld : bool
            _
        pkrot : bool
            _
        all_sources : bool
            _
        fan_inlet_source : bool
            _
        fan_discharge_source : bool
            _
        core_source : bool
            _
        jet_mixing_source : bool
            _
        jet_shock_source : bool
            _
        airframe_source : bool
            _
        fan_igv : bool
            _
        fan_id : bool
            _
        fan_combination_tones : bool
            _
        fan_liner_suppression: bool
            _
        airframe_hsr_calibration : bool
            _
        direct_propagation : bool
            _
        absorption : bool 
            _
        ground_effects : bool
            _
        lateral_attenuation : bool
            _
        shielding : bool
            _
        tones_under_800Hz : bool
            _
        epnl_bandshare : bool
            _
        core_jet_suppression : bool
            _
        save_results : bool
            _
        verification : bool
            _
        F00 : float
            _
        z_cb : float
            _
        v_max : float
            _
        x_max : float
            _
        epnl_dt : float
            _
        atmosphere_dT : float
            _
        max_iter : int
            _
        tolerance : float
            _
        x_microphones : np.ndarray
            _
        ground_resistance : float
            _
        incoherence_constant : float
            _
        n_frequency_bands : int
            _
        n_frequency_subbands : int
            _
        n_altitude_absorption: int
            _
        n_harmonics : int
            _
        n_shock : int
            _
        A_e : float
            _
        r_0 : float
            _
        p_ref : float
            _
        """

        self.settings = dict()
        self.settings['case_name'] = case_name
        self.settings['aircraft_name'] = aircraft_name
        self.settings['engine_name'] = engine_name
        self.settings['output_directory_name'] = output_directory_name
        self.settings['output_file_name'] = output_file_name
        self.settings['time_history_file_name'] = time_history_file_name
        self.settings['engine_deck_file_name'] = engine_deck_file_name
        self.settings['atmosphere_mode'] = atmosphere_mode
        self.settings['fan_BB_method'] = fan_BB_method
        self.settings['fan_RS_method'] = fan_RS_method
        self.settings['fan_ge_flight_cleanup'] = fan_ge_flight_cleanup
        self.settings['core_turbine_attenuation_method'] = core_turbine_attenuation_method
        self.settings['lateral_attenuation_engine_mounting'] = lateral_attenuation_engine_mounting
        self.settings['levels_int_metric'] = levels_int_metric
        self.settings['observer_lst'] = observer_lst
        self.settings['trajectory_mode'] = trajectory_mode
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
        self.settings['x_microphones'] = x_microphones
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

        # Initialize aircraft
        self.aircraft = Aircraft(settings=self.settings)
        if self.settings['trajectory_mode'] == 'model':
            self.aircraft.get_aerodynamics_deck(settings=self.settings)
            self.aircraft.engine.get_performance_deck(settings=self.settings)

        # Initialize pyNA model
        self.problem = Problem()

        # Trajectory model
        if self.settings['trajectory_mode'] == 'model':
            self.trajectory = Trajectory(problem=self.problem, settings=self.settings, aircraft=self.aircraft)
            # TO DO: assess size of the output trajectory

        elif self.settings['trajectory_mode'] == 'time_history':
            self.trajectory = TimeHistory(problem=self.problem, settings=self.settings)
            self.trajectory.load_data()

        # Noise model
        if self.settings['all_sources']:
            self.settings['fan_inlet_source'] = True
            self.settings['fan_discharge_source'] = True
            self.settings['core_source'] = True
            self.settings['jet_mixing_source'] = True
            self.settings['jet_shock_source'] = False
            self.settings['airframe_source'] = True

            if self.settings['verification']:
                self.stca_data = StcaData(settings=self.settings)

        if os.environ['PYNA_LANGUAGE'] == 'python': 
            self.noise = Noise(problem=self.problem, settings=self.settings, aircraft=self.aircraft, trajectory=self.trajectory)
        elif os.environ['PYNA_LANGUAGE'] == 'julia':
            self.noise = NoiseJl(problem=self.problem, settings=self.settings, aircraft=self.aircraft, trajectory=self.trajectory)
        else:
            raise ValueError('Invalid PYNA_LANGUAGE environment variable. Specify: "python" or "julia".')

        return

    def calculate_time_history(self):

        """
        
        Parameters 
        ----------
        trajectory : 
            _
        settings : 
            _

        """

        # OpenMDAO problem setup
        self.problem.setup(force_alloc_complex=True)
        
        # Set trajectory initial conditions
        self.trajectory.set_initial_conditions()

        # OpenMDAO run model
        self.problem.run_model()

        pass

    def calculate_trajectory(self, objective='time', path_init=None) -> None:
        
        """
        """

        self.problem.set_objective(objective=objective)
        self.problem.set_driver_settings(settings=self.settings, objective=objective)
        self.problem.setup(force_alloc_complex=True)
        self.trajectory.set_initial_conditions(path_init=path_init)

        # Attach a recorder to the problem to save model data
        if self.settings['save_results']:
            self.add_recorder(om.SqliteRecorder(pyNA.__path__.__dict__["_path"][0] + '/cases/' + self.settings['case_name'] + '/output/' + self.settings['output_directory_name'] + '/' + self.settings['output_file_name']))

        # Run problem
        dm.run_problem(self.problem, run_driver=True)

        # Save the results
        if self.settings['save_results']:
            self.record(case_name=self.settings['case_name'])

        # Check convergence
        self.problem.check_convergence(settings=self.settings, filename='IPOPT_trajectory_convergence.out')
        
        return None

    def calculate_noise():

        return

    def plot_trajectory(self, path_compare=[], labels_compare=[]):

        """
        
        :param path_compare:
        :type path_compare: tuple of om.Problem()
        """

        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use(pyNA.__path__.__dict__["_path"][0] + '/src/' + 'plot.mplstyle')

        ax[0,0].plot(self.problem.get_val('trajectory.x'), self.problem.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        for i,path in enumerate(path_compare):
            ax[0,0].plot(path.get_val('trajectory.x'), path.get_val('trajectory.z'), '-', label=labels_compare[i])
        ax[0,0].set_xlabel('x [m]')
        ax[0,0].set_ylabel('z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.problem.get_val('trajectory.t_s'), self.problem.get_val('trajectory.v'), '-', color='k')
        for path in path_compare:
            ax[0,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.v'), '-')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.problem.get_val('trajectory.t_s'), self.problem.get_val('trajectory.gamma'), '-', color='k')
        for path in path_compare:
            ax[0,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.gamma'), '-')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.problem.get_val('trajectory.t_s'), 1 / 1000. * self.problem.get_val('trajectory.F_n'), '-', color='k')
        for path in path_compare:
            ax[1,0].plot(path.get_val('trajectory.t_s'), 1 / 1000. * path.get_val('trajectory.F_n'), '-')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.problem.get_val('trajectory.t_s'), self.problem.get_val('trajectory.tau'), '-', color='k')
        for path in path_compare:
            ax[1,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.tau'), '-')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$\tau$ [-]')
        ax[1,1].set_ylim([0,1.02])
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.problem.get_val('trajectory.t_s'), self.problem.get_val('trajectory.alpha'), '-', color='k')
        for path in path_compare:
            ax[1,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.alpha'), '-')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None
    
    def plot_noise_timeseries(self, metric='pnlt') -> None:

        """

        Plot the noise metric along the trajectory.

        Arguments
        ---------
        metric: str
            noise metric to plot. Specify 'pnlt' or 'oaspl'/

        """

        # Create figure
        fig, ax = plt.subplots(1, len(self.settings['observer_lst']), figsize=(20, 4.3), dpi=100)
        if len(self.settings['observer_lst']) == 1:
            ax = [ax]
        ax_zoom = copy.copy(ax)
        plt.style.use(pyNA.__path__.__dict__["_path"][0] + '/src/' + 'plot.mplstyle')

        colors = plt.cm.magma(np.linspace(0,0.8,2))

        # Iterate over observer locations
        for i, observer in enumerate(self.settings['observer_lst']):

            # Time range of epnl domain of dependence
            time_epnl = self.problem.get_val('noise.t_o')[i,:][np.where(self.problem.get_val('noise.'+metric)[i,:] > max(self.problem.get_val('noise.'+metric)[i,:]) - 10.)]

            if metric == 'pnlt':
                if observer == 'lateral':
                    ax[i].plot(self.problem.get_val('noise.t_o')[i,:180], self.problem.get_val('noise.'+metric)[i,:180], linewidth=2.5, label='pyNA', color=colors[0])
                else:
                    ax[i].plot(self.problem.get_val('noise.t_o')[i,:], self.problem.get_val('noise.'+metric)[i,:], linewidth=2.5, label='pyNA', color=colors[0])                
                ax[i].fill_between([time_epnl[0], time_epnl[-1]], [-5, -5], [110., 110.], alpha=0.15, color=colors[0])
                ax[i].set_ylabel('$PNLT_{' + observer + '}$ [TPNdB]')

                # Zoomed-in plots
                ax_zoom[i] = zoomed_inset_axes(ax[i], zoom=4, loc='lower right')
                ax_zoom[i].plot(self.problem.get_val('noise.t_o')[i,:180], self.problem.get_val('noise.'+metric)[i,:180], linewidth=2.5, color=colors[0])
                ax_zoom[i].set_xticks([])
                ax_zoom[i].set_yticks([])

                ax_zoom[i].set_xlim([time_epnl[0], time_epnl[-1]])
                ax_zoom[i].set_ylim([np.max(self.problem.get_val('noise.'+metric)[i,:])-11, np.max(self.problem.get_val('noise.'+metric)[i,:])+1.5])
                mark_inset(ax[i], ax_zoom[i], loc1=1, loc2=3)                

                if self.settings['verification']:
                    self.stca_data.load_levels_time_history()
                    ax[i].plot(self.stca_data.levels_time_history[observer]['t observer [s]'], self.stca_data.levels_time_history[observer]['PNLT'], '--', linewidth=2.5, label='NASA STCA (Berton et al.)', color=colors[1])
                    ax_zoom[i].plot(self.stca_data.levels_time_history[observer]['t observer [s]'], self.stca_data.levels_time_history[observer]['PNLT'], '--', linewidth=2.5, color=colors[1])

            elif metric == 'oaspl':
                ax[i].plot(self.problem.get_val('noise.t_o')[i,:], self.problem.get_val('noise.'+metric)[i,:], linewidth=2.5, label='pyNA')
                ax[i].set_ylabel('$OASPL_{' + observer + '}$ [dB]')

                if self.settings['verification']:
                    self.stca_data.load_trajectory_verification_data()
                    ax[i].plot(self.stca_data.levels_time_history[observer]['t observer [s]'], self.stca_data.levels_time_history[observer]['OASPL'], '--', linewidth=2.5, label='NASA STCA (Berton et al.)')

            ax[i].grid(False)
            ax[i].set_xlabel('Time after brake release [s]')
            ax[i].tick_params(axis='both')
            ax[i].set_ylim([-5, 110])
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
        
        # Set legend
        ax[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.09), ncol=1, borderaxespad=0, frameon=False)
        plt.subplots_adjust(wspace=0.2)
        plt.show()



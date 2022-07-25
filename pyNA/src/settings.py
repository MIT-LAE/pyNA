from ast import Not
from dataclasses import dataclass
import numpy as np
import os
import pdb

class FrozenClass(object):
    __is_frozen = False
    def __setattr__(self, key, value):
        if self.__is_frozen and not hasattr(self, key):
            raise TypeError( "Invalid attribute specified for the %r class." % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__is_frozen = True

@dataclass()
class Settings(FrozenClass):
    """
    pyNA settings class
    """

    def __init__(self, case_name, 
                language = 'python', 
                pyNA_directory = '.', 
                engine_file_name = 'Engine_to.csv', 
                trajectory_file_name = 'Trajectory_to.csv',
                output_directory_name = '',
                output_file_name = 'Trajectory_stca.sql', 
                ac_name = 'stca', 
                ac_version = '',
                save_results = False, 
                fan_inlet = True, 
                fan_discharge = True,
                core = True, 
                jet_mixing = True, 
                jet_shock = False, 
                airframe = True, 
                all_sources = False,
                fan_igv = False,
                fan_id = False, 
                observer_lst = ('lateral', 'flyover'),
                method_core_turb ='GE', 
                fan_BB_method ='geae', 
                fan_RS_method = 'allied_signal',
                ge_flight_cleanup = 'takeoff', 
                levels_int_metric = 'epnl', 
                engine_mounting = 'none', 
                direct_propagation = True,
                absorption = True, 
                groundeffects = True, 
                lateral_attenuation = True, 
                suppression = True, 
                fan_liner_suppression = True,
                shielding = False, 
                hsr_calibration = True, 
                validation = False, 
                bandshare = False, 
                TCF800 = True, 
                combination_tones = False,
                N_shock = 8, 
                dT = 10.0169, 
                sigma = 291.0 * 515.379, 
                a_coh = 0.01, 
                N_f = 24, 
                N_b = 5, 
                n_altitude_absorption = 5,
                A_e = 10.334 * (0.3048 ** 2), 
                dt_epnl = 0.5, 
                n_harmonics = 10, 
                r_0 = 0.3048, 
                p_ref= 2e-5, 
                x_observer_array = np.array([[12325.*0.3048, 450., 4*0.3048], [21325.*0.3048, 0., 4*0.3048]]),
                noise_optimization = False, 
                noise_constraint_lateral = 200.,
                PTCB = False, 
                PHLD = False, 
                PKROT = False,
                TS_to = 1.0, 
                TS_vnrs = 1.0, 
                TS_cutback = None, 
                z_cutback = 500., 
                theta_flaps = 10.,
                theta_slats = -6., 
                n_order = 3,
                max_iter = 200, 
                tol = 1e-5,
                Foo = 83821.6):

        """
        Initialize pyNA settings class

        :param case_name: Case name [-]
        :type case_name: str
        :param pyNA_directory: Directory where pyNA is installed
        :type pyNA_directory: str
        :param engine_file_name: File name of the take-off engine inputs [-]
        :type engine_file_name: str
        :param trajectory_file_name: File name of the take-off trajectory [-]
        :type trajectory_file_name: str
        :param output_directory_name: Name of the directory of output .sql file [-]
        :type output_directory_name: str
        :param output_file_name: Name of the output .sql file [-]
        :type output_file_name: str
        :param ac_name: Name of the aircraft [-]
        :type ac_name: str
        :param save_results: Flag to save results [-]
        :type save_results: bool
        :param fan_inlet: Enable fan inlet noise source [-]
        :type fan_inlet: bool
        :param fan_discharge: Enable fan discharge noise source [-]
        :type fan_discharge: bool
        :param core: Enable core noise source [-]
        :type core: bool
        :param jet_mixing: Enable jet mixing noise source [-]
        :type jet_mixing: bool
        :param jet_shock: Enable jet shock noise source [-]
        :type jet_shock: bool
        :param airframe: Enable airframe noise source [-]
        :type airframe: bool
        :param all_sources: Enable all noise sources [-]
        :type all_sources: bool
        :param trajectory_mode: mode for trajectory calculations [-] ('cutback', optimization')
        :type trajectory_mode: str
        :param observer_lst: List of observers to analyze [-] ('flyover','lateral','approach', 'contour')
        :type observer_lst: lst
        :param method_core_turb: Method to account for turbine transmission in the combustor ('GE', 'PW') [-]
        :type method_core_turb: str
        :param fan_BB_method: Method BB (original / allied_signal / geae / kresja) [-]
        :type fan_BB_method: str
        :param fan_RS_method: Method RS (original / allied_signal / geae / kresja) [-]
        :type fan_RS_method: str
        :param fan_igv: Enable fan inlet guide vanes
        :type fan_igv: bool
        :param fan_id: Enable fan inlet distortions
        :type fan_id: bool
        :param ge_flight_cleanup: GE flight cleanup switch (none/takeoff/approach) [-]
        :type ge_flight_cleanup: str
        :param levels_int_metric: Integrated noise metric [-]
        :type levels_int_metric: str
        :param engine_mounting: Engine mounting ('fuselage'/'underwing'/'none') [-]
        :type engine_mounting: str
        :param direct_propagation: Flag for direct propagation (including distance and characteristic impedance effects) [-]
        :type direct_propagation: bool
        :param absorption: Flag for atmospheric absorption [-]
        :type absorption: bool
        :param groundeffects: Flag for ground effects [-]
        :type groundeffects: bool
        :param lateral_attenuation: Flag for empirical lateral attenuation effects [-]
        :type lateral_attenuation: bool
        :param suppression: Flag for suppression of engine modules [-]
        :type suppression: bool
        :param fan_liner_suppression: Flag for fan liner suppression [-]
        :type fan_liner_suppression: bool
        :param shielding: Flag for shielding effects (not implemented yet) [-]
        :type shielding: bool
        :param hsr_calibration: Flag for HSR-era airframe calibration [-]
        :type hsr_calibration: bool
        :param validation: Flag for validation with NASA STCA noise model [-]
        :type validation: bool
        :param bandshare: Flag to plot PNLT [-]
        :type bandshare: bool
        :param TCF800: Flag for tone penalty addition to PNLT metric; allows any tone below 800Hz to be ignored [-]
        :type TCF800: bool
        :param combination_tones: Flag for combination tones int he fan noise model [-]
        :type combination_tones: bool
        :param N_shock: Number of shocks in supersonic jet [-]
        :type N_shock: int
        :param dT: dT standard atmosphere [K]
        :type dT: float
        :param sigma: Specific flow resistance of ground [kg/s m3]
        :type sigma: float
        :param a_coh: Incoherence constant [-]
        :type a_coh: float
        :param N_f: Number of discrete 1/3 octave frequency bands [-]
        :type N_f: int
        :param N_b: Number of bands (propagation) [-]
        :type N_b: int
        :param n_altitude_absorption: Number of integration steps in atmospheric propagation [-]
        :type n_altitude_absorption: int
        :param A_e: Engine reference area [m2]
        :type A_e: float
        :param dt_epnl: Time step of to calculate EPNL from interpolated PNLT data [s]
        :type dt_epnl: float
        :param n_harmonics: Number of harmonics to be considered in tones [-]
        :type n_harmonics: int
        :param r_0: Distance source observer in source mode [m]
        :type r_0: float
        :param p_ref: Reference pressure [Pa]
        :type p_ref: float
        :param noise_optimization: Flag to noise-optimize the trajectory [-]
        :type noise_optimization: bool
        :param noise_constraint_lateral: Constraint on the lateral noise [EPNdB]
        :type noise_constraint_lateral: float
        :param PTCB: Enable PTCB [-]
        :type PTCB: bool
        :param PHLD: Enable PHLD [-]
        :type PHLD: bool
        :param PKROT: Enable PKROT [-]
        :type PKROT: bool
        :param TS_to: Engine TO thrust-setting (values < 1 denote APR) [-]
        :type TS_to: float
        :param TS_vnrs: Engine VNRS thrust-setting [-]
        :type TS_vnrs: float
        :param TS_cutback: Engine cutback thrust-setting [-]
        :type TS_cutback: float
        :param z_cutback: z-location of cutback [m]
        :type z_cutback: float
        :param theta_flaps: Flap deflection angles [deg]
        :type theta_flaps: float
        :param theta_slats: Slat deflection angles [deg]
        :type theta_slats: float
        :param max_iter: Maximum number of iterations for trajectory computations [-]
        :type max_iter: int
        :param tol: Tolerance for trajectory computations [-]
        :type tol: float
        :param Foo: sea level static thrust [N]
        :type Foo: float

        """

        self.case_name = case_name
        self.language = language
        self.pyNA_directory = pyNA_directory
        self.engine_file_name = engine_file_name
        self.trajectory_file_name = trajectory_file_name
        self.output_directory_name = output_directory_name
        self.output_file_name = output_file_name
        self.ac_name = ac_name
        self.ac_version = ac_version

        self.save_results = save_results

        self.fan_inlet = fan_inlet
        self.fan_discharge = fan_discharge
        self.core = core
        self.jet_mixing = jet_mixing
        self.jet_shock = jet_shock
        self.airframe = airframe
        self.all_sources = all_sources

        self.observer_lst = observer_lst
        self.x_observer_array = x_observer_array
        self.method_core_turb = method_core_turb
        self.fan_BB_method = fan_BB_method
        self.fan_RS_method = fan_RS_method
        self.fan_igv = fan_igv
        self.fan_id = fan_id
        self.ge_flight_cleanup = ge_flight_cleanup
        self.levels_int_metric = levels_int_metric
        self.engine_mounting = engine_mounting

        self.direct_propagation = direct_propagation
        self.absorption = absorption
        self.groundeffects = groundeffects
        self.lateral_attenuation = lateral_attenuation
        self.suppression = suppression
        self.fan_liner_suppression = fan_liner_suppression
        self.shielding = shielding
        self.hsr_calibration = hsr_calibration
        self.validation = validation
        self.bandshare = bandshare
        self.TCF800 = TCF800
        self.combination_tones = combination_tones

        self.N_shock = N_shock
        self.dT = dT
        self.sigma = sigma
        self.a_coh = a_coh
        self.N_f = N_f
        self.N_b = N_b
        self.n_altitude_absorption = n_altitude_absorption
        self.A_e = A_e
        self.dt_epnl = dt_epnl
        self.n_harmonics = n_harmonics
        self.r_0 = r_0
        self.p_ref = p_ref

        self.noise_optimization = noise_optimization
        self.noise_constraint_lateral = noise_constraint_lateral
        self.PTCB = PTCB
        self.PHLD = PHLD
        self.PKROT = PKROT
        self.TS_to = TS_to
        self.TS_vnrs = TS_vnrs
        self.TS_cutback = TS_cutback
        self.z_cutback = z_cutback
        self.theta_flaps = theta_flaps
        self.theta_slats = theta_slats
        self.n_order = n_order
        self.max_iter = max_iter
        self.tol = tol

        self.Foo = Foo

        # Freeze self.settings
        self._freeze()

    def check(self) -> None:

        """
        Check the pyNA settings before a run.

        :return: None
        """

        # pyNA directory
        if type(self.pyNA_directory) != str:
            raise TypeError(self.pyNA_directory, "is not a valid directory location. Specify the name (string).")

        # Folder and file names
        if type(self.case_name) != str:
            raise TypeError(self.case_name, "does not have the correct type for the engine file name. type(settings.case_name) must be str.")
        if type(self.engine_file_name) != str:
            raise TypeError(self.engine_file_name, "does not have the correct type for the engine file name. type(settings.engine_file_name) must be str.")
        if type(self.trajectory_file_name) != str:
            raise TypeError(self.trajectory_file_name, "does not have the correct type for the trajectory file name. type(settings.trajectory_file_name) must be str.")
        if type(self.output_directory_name) != str:
            raise TypeError(self.output_directory_name, "does not have the correct type for the output file name. type(settings.output_directory_name) must be str.")
        if type(self.output_file_name) != str:
            raise TypeError(self.output_file_name, "does not have the correct type for the output file name. type(settings.output_file_name) must be str.")
        if self.ac_name not in ['stca', 'a10']:
            raise TypeError(self.ac_name, "is not a valid aircraft name. Specify: 'stca', 'stca_verification', 'a10'.")
        if type(self.ac_version) != str:
            raise TypeError(self.ac_version, "does not have the correct type for the aircraft version. type (self.ac_version) must be str.")

        # Flags
        if type(self.save_results) != bool:
            raise TypeError(self.save_results, "does not have the correct type. type(settings.save_results) must be bool.")
        if type(self.fan_inlet) != bool:
            raise TypeError(self.fan_inlet, "does not have the correct type. type(settings.fan_inlet) must be bool.")
        if type(self.fan_discharge) != bool:
            raise TypeError(self.fan_discharge, "does not have the correct type. type(settings.fan_discharge) must be bool.")
        if type(self.core) != bool:
            raise TypeError(self.core, "does not have the correct type. type(settings.core) must be bool.")
        if type(self.jet_mixing) != bool:
            raise TypeError(self.jet_mixing, "does not have the correct type. type(settings.jet_mixing) must be bool.")
        if type(self.jet_shock) != bool:
            raise TypeError(self.jet_shock, "does not have the correct type. type(settings.jet_shock) must be bool.")
        if type(self.airframe) != bool:
            raise TypeError(self.airframe, "does not have the correct type. type(settings.airframe) must be bool.")
        if type(self.all_sources) != bool:
            raise TypeError(self.all_sources, "does not have the correct type. type(settings.all_sources) must be bool.")
        if type(self.fan_igv) != bool:
            raise TypeError(self.fan_igv, "does not have the correct type. type(settings.fan_igv) must be bool.")
        if type(self.fan_id) != bool:
            raise TypeError(self.fan_id, "does not have the correct type. type(settings.fan_id) must be bool.")
        if type(self.direct_propagation) != bool:
            raise TypeError(self.direct_propagation, "does not have the correct type. type(settings.direct_propagation) must be bool.")
        if type(self.absorption) != bool:
            raise TypeError(self.absorption, "does not have the correct type. type(settings.absorption) must be bool.")
        if type(self.groundeffects) != bool:
            raise TypeError(self.groundeffects, "does not have the correct type. type(settings.groundeffects) must be bool.")
        if type(self.lateral_attenuation) != bool:
            raise TypeError(self.lateral_attenuation, "does not have the correct type. type(settingslateral_attenuation) must be bool.")
        if type(self.suppression) != bool:
            raise TypeError(self.suppression, "does not have the correct type. type(settings.suppression) must be bool.")
        if type(self.fan_liner_suppression) != bool:
            raise TypeError(self.fan_liner_suppression, "does not have the correct type. type(settings.fan_liner_suppression) must be bool.")
        if type(self.shielding) != bool:
            raise TypeError(self.shielding, "does not have the correct type. type(settings.shielding) must be bool.")
        if type(self.hsr_calibration) != bool:
            raise TypeError(self.hsr_calibration, "does not have the correct type. type(settings.hsr_calibration) must be bool.")
        if type(self.validation) != bool:
            raise TypeError(self.validation, "does not have the correct type. type(settings.validation) must be bool.")
        if type(self.bandshare) != bool:
            raise TypeError(self.bandshare, "does not have the correct type. type(settings.bandshare) must be bool.")
        if type(self.TCF800) != bool:
            raise TypeError(self.TCF800, "does not have the correct type. type(settings.TCF800) must be bool.")
        if type(self.combination_tones) != bool:
            raise TypeError(self.combination_tones, "does not have the correct type. type(settings.combination_tones) must be bool.")
        if type(self.noise_optimization) != bool:
            raise TypeError(self.noise_optimization, "does not have the correct type. type(settings.noise_optimization) must be bool.")
        if type(self.noise_constraint_lateral) not in [float, np.float32, np.float64]:
            raise TypeError(self.noise_constraint_lateral, "does not have the correct type. type(settings.noise_constraint_lateral) must be [float, np.float32, np.float64]")

        if type(self.PTCB) != bool:
            raise TypeError(self.PTCB, "does not have the correct type. type(settings.PTCB) must be bool.")
        if type(self.PHLD) != bool:
            raise TypeError(self.PHLD, "does not have the correct type. type(settings.PHLD) must be bool.")
        if type(self.PKROT) != bool:
            raise TypeError(self.PKROT, "does not have the correct type. type(settings.PKROT) must be bool.")

        # Methods
        if self.method_core_turb not in ['GE', 'PW']:
            raise ValueError(self.method_core_turb, "is not a valid option for the core turbine attenuation method. Specify: 'GE'/'PW'.")
        if self.fan_BB_method not in ['original', 'allied_signal', 'geae', 'kresja']:
            raise ValueError(self.fan_BB_method, "is not a valid option for the fan broadband method. Specify: 'original'/'allied_signal'/'geae'/'kresja'.")
        if self.fan_RS_method not in ['original', 'allied_signal', 'geae', 'kresja']:
            raise ValueError(self.fan_RS_method, "is not a valid option for the fan rotor-stator interation method. Specify: 'original'/'allied_signal'/'geae'/'kresja'.")
        if self.ge_flight_cleanup not in ['none', 'takeoff', 'approach']:
            raise ValueError(self.ge_flight_cleanup, "is not a valid option for the GE flight clean-up effects method. Specify: 'none'/'takeoff'/'approach'.")
        if self.levels_int_metric not in ['epnl', 'ipnlt', 'ioaspl', 'sel']:
            raise ValueError(self.levels_int_metric, "is not a valid option for the integrated noise levels metric. Specify: 'epnl'/'ipnlt'/'ioaspl'/'sel'.")
        if self.engine_mounting not in ['fuselage', 'underwing', 'none']:
            raise ValueError(self.engine_mounting, "is not a valid option for the engine mounting description. Specify: 'fuselage', 'underwing', 'none'.")

        # Values
        if type(self.N_shock) not in [int, np.int32, np.int64]:
            raise TypeError(self.N_shock, "does not have the correct type. type(settings.N_shock) must be [int, np.int32, np.int64]")
        if type(self.dT) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.dT, "does not have the correct type. type(settings.dT) must be [float, np.float32, np.float64]")
        if type(self.sigma) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.sigma, "does not have the correct type. type(settings.sigma) must be [float, np.float32, np.float64]")
        if type(self.a_coh) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.a_coh, "does not have the correct type. type(settings.a_coh) must be [float, np.float32, np.float64]")
        if type(self.N_f) not in [int, np.int32, np.int64]:
            raise TypeError(self.N_f, "does not have the correct type. type(settings.N_f) must be [int, np.int32, np.int64]")
        if type(self.N_b) not in [int, np.int32, np.int64]:
            raise TypeError(self.N_b, "does not have the correct type. type(settings.N_b) must be [int, np.int32, np.int64]")
        if np.remainder(self.N_b, 2) != 1:
            raise ValueError("The number of 1/3rd octave frequency sub-bands needs to be odd.")
        if type(self.n_altitude_absorption) not in [int, np.int32, np.int64]:
            raise TypeError(self.n_altitude_absorption, "does not have the correct type. type(settings.n_altitude_absorption) must be [int, np.int32, np.int64]")
        if type(self.n_harmonics) not in [int, np.int32, np.int64]:
            raise TypeError(self.n_harmonics, "does not have the correct type. type(settings.n_harmonics) must be [int, np.int32, np.int64]")
        if type(self.A_e) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.A_e, "does not have the correct type. type(settings.A_e) must be [float, np.float32, np.float64]")
        if type(self.dt_epnl) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.dt_epnl, "does not have the correct type. type(settings.dt_epnl) must be [float, np.float32, np.float64]")
        if type(self.r_0) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.r_0, "does not have the correct type. type(settings.r_0) must be [float, np.float32, np.float64]")
        if type(self.p_ref) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.p_ref, "does not have the correct type. type(settings.p_ref) must be [float, np.float32, np.float64]")

        if type(self.x_observer_array) != np.ndarray:
            raise TypeError(self.x_observer_array, "does not have the correct type. type(settings.x_observer_array) must be np.ndarray")
        if self.observer_lst in [('lateral', ), ('flyover', )] and np.shape(self.x_observer_array) != (1,3):
                raise ValueError("Shape of the x_observer_array must be (1, 3); instead shape is ", np.shape(self.x_observer_array))
        elif self.observer_lst in [('lateral', 'flyover', ), ('flyover', 'lateral')] and np.shape(self.x_observer_array) != (2,3):
                raise ValueError("Shape of the x_observer_array must be (2, 3); instead shape is ", np.shape(self.x_observer_array))

        # Trajectory options
        if type(self.TS_to) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.TS_to, "does not have the correct type. type(settings.TS_to) must be in [float, np.float32, np.float64]")
        if type(self.TS_vnrs) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.TS_vnrs, "does not have the correct type. type(settings.TS_vnrs) must be in [float, np.float32, np.float64]")
        NoneType = type(None)
        if type(self.TS_cutback) not in [NoneType, float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.TS_cutback, "does not have the correct type. type(settings.TS_cutback) must be in [float, np.float32, np.float64]")
        if type(self.z_cutback) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.z_cutback, "does not have the correct type. type(settings.z_cutback) must be in [float, np.float32, np.float64]")
        if type(self.theta_flaps) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.theta_flaps, "does not have the correct type. type(settings.theta_flaps) must be np.ndarray")
        if type(self.theta_slats) not in [float, np.float32, np.float64, int, np.int32, np.int64]:
            raise TypeError(self.theta_slats, "does not have the correct type. type(settings.theta_slats) must be np.ndarray")
        if type(self.n_order) not in [int, np.int32, np.int64]:
            raise TypeError(self.n_order, "does not have the correct type. type(settings.n_order) must be in [int, np.int32, np.int64]")
        if type(self.max_iter) not in [int, np.int32, np.int64]:
            raise TypeError(self.max_iter, "does not have the correct type. type(settings.max_iter) must be in [int, np.int32, np.int64]")
        if type(self.tol) not in [float, np.float32, np.float64]:
            raise TypeError(self.tol, "does not have the correct type. type(settings.tol) must be in [float, np.float32, np.float64]")

        # Observer list
        for observer in self.observer_lst:
            if observer not in ['lateral','flyover', 'approach', 'contours', 'optimization']:
                raise ValueError(observer, "is not a valid option for the observer list. Specify any from 'lateral'/'flyover'/'approach'/'contours'/'optimization'")

        # Language to use to solve components (julia/python)
        if self.language not in ['python', 'julia']:
            raise ValueError("Invalid environment variable pyna_language. Specify 'python'/'julia'.")

        # Set all noise components equal to True if settings.all_sources == True
        if self.all_sources:
            self.fan_inlet = True
            self.fan_discharge = True
            self.core = True
            self.jet_mixing = True
            self.jet_shock = True
            self.airframe = True

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
            self.validation = False

        return
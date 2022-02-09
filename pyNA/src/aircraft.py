import pdb
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pyNA.src.settings import Settings
import scipy

@dataclass
class Aircraft:
    """
    Aircraft class containing vehicle constants and aerodynamics data.

    """

    # Vehicle parameters
    mtow : float                # Max. take-off weight [kg]
    n_eng : int                 # Number of engines installed on the aircraft [-]
    comp_lst : list             # List of airframe components to include in noise analysis [-]

    # Airframe parameters
    af_S_h : float              # Horizontal tail area (0.0 ft2) [m2]
    af_S_v : float              # Vertical tail area (361.8 ft2) [m2]
    af_S_w : float              # Wing area (3863.0 ft2) [m2]
    af_b_f : float              # Flap span (25.74 ft) [m]
    af_b_h : float              # Horizontal tail span (0. ft) [m]
    af_b_v : float              # Vertical tail span (57.17 ft) [m]
    af_b_w : float              # Wing span (94.43 ft) [m]

    # High-lift devices
    theta_flaps : float         # Flap deflection angle [deg]
    theta_slats: float          # Slat deflection angle [deg]
    af_S_f : float              # Flap area (120.0 ft2) [m2]
    af_s : int                  # Number of slots for trailing-edge flaps (min. is 1) [-]

    # Landing gear
    af_d_mg : float             # Tire diameter of main landing gear (3.08 ft) [m]
    af_d_ng : float             # Tire diameter of nose landing gear (3.71 ft) [m]
    af_l_mg : float             # Main landing-gear strut length (7.5 ft) [m]
    af_l_ng : float             # Nose landing-gear strut length (6.0 ft) [m]
    af_n_mg : float             # Number of wheels per main landing gear [-]
    af_n_ng : float             # Number of wheels per nose landing gear [-]
    af_N_mg : float             # Number of main landing gear [-]
    af_N_ng : float             # Number of nose landing gear [-]
    mu_r : float                # Rolling resistance coefficient [-]

    # Engine parameters
    B_fan : int                # Number of fan blades [-]
    V_fan : int                # Number of fan vanes [-]
    RSS_fan : float            # Rotor-stator spacing [%]
    M_d_fan : float            # Relative tip Mach number of fan at design [-]

    inc_F_n : float            # Thrust inclination angle [deg]
    TS_lower : float           # Min. power setting [-]
    TS_upper : float           # Max. power setting [-]

    # Airframe configuration
    af_clean_w : bool          # Clean wing (1: yes / 0: no)
    af_clean_h : bool          # Clean horizontal tail (1: yes / 0: no)
    af_clean_v: bool           # Clean vertical tail (1: yes / 0: no)
    af_delta_wing : bool       # Delta wing (1: yes / 0: no)

    # Aerodynamics and flight performance
    c_l_max : float             # Max. lift coefficient [-]
    alpha_0 : float            # Wing mounting angle [deg]
    k_rot : float              # Rotation coefficient (v_rot/v_stall) [-]
    v_max : float              # Maximum climb-out velocity [m/s]
    z_max: float               # Maximum climb-out altitude [m]
    gamma_rot : float          # Initial climb angle [deg]

    def __init__(self, name: str, version: str, settings: Settings) -> None:
        """
        Initialize Aircraft class.

        :param name: aircraft name
        :type name: str
        :param version: aircraft version
        :type version: str
        :param settings: pyna settings
        :type settings: Settings

        :return: None
        """

        # Initialize aircraft name
        self.name = name
        self.version = version

        # Initialize aerodynamics deck and flight performance parameters
        self.aero = dict()
        # self.v_stall: float
        # self.v_rot: float

        # Load aircraft parameters
        path = settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/' + self.name + '_' + self.version + '.json'
        with open(path) as f:
            params = json.load(f)
        Aircraft.set_aircraft_parameters(self, **params)

    def set_aircraft_parameters(self, mtow: np.float64, n_eng: np.int64, comp_lst: list, af_S_h: np.float64,
                                af_S_v: np.float64, af_S_w: np.float64, af_b_f: np.float64, af_b_h: np.float64, af_b_v:
                                np.float64, af_b_w: np.float64, theta_flaps: np.float64, theta_slats: np.float64,
                                af_S_f: np.float64, af_s: np.int64, af_d_mg: np.float64, af_d_ng: np.float64, af_l_mg:
                                np.float64, af_l_ng: np.float64, af_n_mg: np.float64, af_n_ng: np.float64, af_N_mg:
                                np.float64, af_N_ng: np.float64, mu_r: np.float64, B_fan: np.int64, V_fan: np.int64,
                                RSS_fan: np.float64, M_d_fan: np.float64, inc_F_n: np.float64, TS_lower: np.float64,
                                TS_upper: np.float64, af_clean_w: bool, af_clean_h:bool, af_clean_v: bool,
                                af_delta_wing: bool, alpha_0: np.float64, k_rot: np.float64, v_max: np.float64, 
                                z_max: np.float64, gamma_rot: np.float64) -> None:
        """
        Set the aircraft parameters in the aircraft class.

        :param mtow: Max. take-off weight [kg]
        :type mtow: np.float64
        :param n_eng: Number of engines installed on the aircraft [-]
        :type n_eng: np.int64
        :param comp_lst: List of airframe components to include [-]
        :type comp_lst: list
        :param af_S_h: Horizontal tail area [m2]
        :type af_S_h: np.float64
        :param af_S_v: Vertical tail area [m2]
        :type af_S_v: np.float64
        :param af_S_w: Wing area [m2]
        :type af_S_w: np.float64
        :param af_b_f: Flap span [m]
        :type af_b_f: np.float64
        :param af_b_h: Horizontal tail span [m]
        :type af_b_h: np.float64
        :param af_b_v: Vertical tail span [m]
        :type af_b_v: np.float64
        :param af_b_w: Wing span [m]
        :type af_b_w: np.float64
        :param theta_flaps: Flap deflection angle [deg]
        :type theta_flaps: np.float64
        :param theta_slats: Slat deflection angle [deg]
        :type theta_slats: np.float64
        :param af_S_f: Flap area [m2]
        :type af_S_f: np.float64
        :param af_s: Number of slots for trailing-edge flaps (min. is 1) [-]
        :type af_s: np.int64
        :param af_d_mg: Tire diameter of main landing gear [m]
        :type af_d_mg: np.float64
        :param af_d_ng: Tire diameter of nose landing gear [m]
        :type af_d_ng: np.float64
        :param af_l_mg: Main landing-gear strut length [m]
        :type af_l_mg: np.float64
        :param af_l_ng: Nose landing-gear strut length [m]
        :type af_l_ng: np.float64
        :param af_n_mg: Number of wheels per main landing gear [-]
        :type af_n_mg: np.int64
        :param af_n_ng: Number of wheels per nose landing gear [-]
        :type af_n_ng: np.int64
        :param af_N_mg: Number of main landing gear [-]
        :type af_N_mg: np.int64
        :param af_N_ng: Number of nose landing gear [-]
        :type af_N_ng: np.int64
        :param mu_r: Rolling resistance coefficient [-]
        :type mu_r: np.float64
        :param B_fan: Number of fan blades [-]
        :type B_fan: np.int64
        :param V_fan: Number of fan vanes [-]
        :type V_fan: np.int64
        :param RSS_fan: Rotor-stator spacing [%]
        :type RSS_fan: np.float64
        :param M_d_fan: Relative tip Mach number of fan at design [-]
        :type M_d_fan: np.float64
        :param inc_F_n: Thrust inclination angle [deg]
        :type inc_F_n: np.float64
        :param TS_lower: Min. power setting [-]
        :type TS_lower: np.float64
        :param TS_upper: Max. power setting [-]
        :type TS_upper: np.float64
        :param af_clean_w: Flag for clean wing configuration [-]
        :type af_clean_w: bool
        :param af_clean_h: Flag for clean horizontal tail configuration [-]
        :type af_clean_h: bool
        :param af_clean_v: Flag for clean vertical tail configuration [-]
        :type af_clean_v: bool
        :param af_delta_wing: Flag for delta wing configuration [-]
        :type af_delta_wing: bool
        :param alpha_0: Wing mounting angle [deg]
        :type alpha_0: np.float64
        :param k_rot: Rotation coefficient (v_rot/v_stall) [-]
        :type k_rot: np.float64
        :param v_max: Maximum climb-out velocity [m/s]
        :type v_max: np.float64
        :param z_max: Maximum climb-out altitude [m]
        :type z_max: np.float64
        :param gamma_rot: Initial climb angle [deg]
        :type gamma_rot: np.float64

        :return: None
        """

        self.mtow = mtow
        self.n_eng = n_eng
        self.comp_lst = comp_lst
        self.af_S_h = af_S_h
        self.af_S_v = af_S_v
        self.af_S_w = af_S_w
        self.af_b_f = af_b_f
        self.af_b_h = af_b_h
        self.af_b_v = af_b_v
        self.af_b_w = af_b_w
        self.theta_flaps = theta_flaps
        self.theta_slats = theta_slats
        self.af_S_f = af_S_f
        self.af_s = af_s
        self.af_d_mg = af_d_mg
        self.af_d_ng = af_d_ng
        self.af_l_mg = af_l_mg
        self.af_l_ng = af_l_ng
        self.af_n_mg = af_n_mg
        self.af_n_ng = af_n_ng
        self.af_N_mg = af_N_mg
        self.af_N_ng = af_N_ng
        self.mu_r = mu_r
        self.B_fan = B_fan
        self.V_fan = V_fan
        self.RSS_fan = RSS_fan
        self.M_d_fan = M_d_fan
        self.inc_F_n = inc_F_n
        self.TS_lower = TS_lower
        self.TS_upper = TS_upper
        self.af_clean_w = af_clean_w
        self.af_clean_h = af_clean_h
        self.af_clean_v = af_clean_v
        self.af_delta_wing = af_delta_wing
        self.alpha_0 = alpha_0
        self.k_rot = k_rot
        self.v_max = v_max
        self.z_max = z_max
        self.gamma_rot = gamma_rot

        return None

    def load_aerodynamics(self, settings: Settings) -> None:
        """
        Load aerodynamic data from aerodynamics deck.

        :param settings: pyNA settings
        :type settings: Settings

        :return: None
        """

        # Load aerodynamics deck
        if settings.ac_name in ['stca', 'stca_verification']:
            # Load data 
            self.aero['alpha'] = np.array([-2.,  0.,  2.,  4.,  6.,  8., 10., 12., 15., 18.])
            self.aero['theta_flaps'] = np.array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20., 22., 24., 26.])
            self.aero['theta_slats'] = np.array([-26., -24., -22., -20., -18., -16., -14., -12., -10.,  -8.,  -6., -4.,  -2.,   0.])
            self.aero['c_l'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_l_stca.npy')
            self.aero['c_l_max'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_l_max_' + settings.ac_name + '.npy')
            self.aero['c_d'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_d_stca.npy')

        elif settings.ac_name == 'a10':
            self.aero['alpha'] = np.array([-2., -1., 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            self.aero['theta_flaps'] = np.array([0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26.])
            self.aero['theta_slats'] = np.array([-26., -24., -22., -20., -18., -16., -14., -12., -10., -8., -6., -4., -2., 0.])
            self.aero['c_l'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_l_' + settings.ac_name + '.npy')
            self.aero['c_l_max'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_l_max_' + settings.ac_name + '.npy')
            self.aero['c_d'] = np.load(settings.pyNA_directory + '/cases/' + settings.case_name + '/aircraft/c_d_' + settings.ac_name + '.npy')
        else:
            raise ValueError('Invalid aircraft name specified.')

        return None

    @staticmethod
    def generate_aerodeck_from_csv(filename_c_l:str, filename_c_d:str, savename_c_l:str=None, savename_cd:str=None) -> None:
        """
        Generate a CL-CD aerodynamics deck from a .csv data file.

        :param filename_c_l: File name with c_l data as function of alpha, theta_flap, theta_slat
        :type filename_c_l: str
        :param filename_cd: File name with cd data as function of alpha, theta_flap, theta_slat
        :type filename_cd: str
        :param savename_c_l: Save file name of c_l-aerodynamics deck
        :type savename_c_l: str
        :param savename_cd: Save file name of c_d-aerodynamics deck
        :type savename_cd: str

        :return: None
        """

        # Load data
        data_c_l = pd.read_csv(filename_c_l)
        data_c_d = pd.read_csv(filename_c_d)

        # Extract angles
        alpha = data_c_l.values[2:, 0]
        theta_slats = np.unique(data_c_l.values[0, 1:])
        theta_flaps = np.unique(data_c_l.values[1, 1:])

        # Extract CL-CD in matrices
        c_l = np.zeros((np.size(alpha), np.size(theta_flaps), np.size(theta_slats)))
        c_d = np.zeros((np.size(alpha), np.size(theta_flaps), np.size(theta_slats)))

        for i in np.arange(np.size(alpha)):
            for j in np.arange(np.size(theta_slats)):
                c_l[i, :, j] = np.flip(
                    data_c_l.values[2 + i, 1 + j * np.size(theta_flaps):1 + (j + 1) * np.size(theta_flaps)])
                c_d[i, :, j] = np.flip(
                    data_c_d.values[2 + i, 1 + j * np.size(theta_flaps):1 + (j + 1) * np.size(theta_flaps)])

        # Save aero deck
        if savename_c_l is not None:
            np.save(savename_c_l, c_l)
        if savename_cd is not None:
            np.save(savename_cd, c_d)

        return None
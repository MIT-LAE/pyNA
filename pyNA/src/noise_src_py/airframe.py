import pdb
import openmdao
import openmdao.api as om
from tqdm import tqdm
import numpy as np
from typing import Dict, Any


class Airframe:

    @staticmethod
    def trailing_edge_wing(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, rho_0: np.float64, mu_0: np.float64, theta: np.float64, phi: np.float64, freq: np.ndarray) -> np.ndarray:
        """
        Compute wing trailing edge mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param rho_0: ambient density [kg/m3]
        :type rho_0: np.float64
        :param mu_0: ambient dynamic viscosity [kg/m/s]
        :type mu_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_w
        :rtype: np.ndarray [n_t, settings.N_f]

        """
        ### ---------------- Wing trailing-edge noise ----------------
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
        delta_w_star = 0.37 * (ac.af_S_w / ac.af_b_w ** 2) * (rho_0 * M_0 * c_0 * ac.af_S_w / (mu_0 * ac.af_b_w)) ** (-0.2)

        # Determine configuration constant and the sound power
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
        if ac.af_clean_w:
            K_w = 7.075e-6
        else:
            K_w = 4.464e-5
        Pi_star_w = K_w * M_0 ** 5 * delta_w_star

        # Determine directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
        D_w = 4. * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

        # Determine spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
        S_w = freq * delta_w_star * ac.af_b_w / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        if ac.af_delta_wing == 1:
            F_w = 0.613 * (10 * S_w) ** 4 * ((10 * S_w) ** 1.35 + 0.5) ** (-4)
        elif ac.af_delta_wing == 0:
            F_w = 0.485 * (10 * S_w) ** 4 * ((10 * S_w) ** 1.5 + 0.5) ** (-4)
        else:
            raise ValueError('Invalid delta-wing flag configuration specified. Specify: 0/1.')

        # Determine msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        r_s_star_af = settings.r_0/ac.af_b_w
        msap_w = 1 / (4 * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (Pi_star_w * D_w * F_w)

        return msap_w

    @staticmethod
    def trailing_edge_horizontal_tail(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, rho_0: np.float64, mu_0: np.float64, theta: np.float64, phi: np.float64, freq: np.ndarray) -> np.ndarray:
        """
        Compute horizontal tail trailing edge mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param rho_0: ambient density [kg/m3]
        :type rho_0: np.float64
        :param mu_0: ambient dynamic viscosity [kg/m/s]
        :type mu_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_h
        :rtype: np.ndarray [settings.N_f]
        """

        # ---------------- Horizontal tail trailing-edge noise ----------------
        # Trailing edge noise of the horizontal tail
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
        delta_h_star = 0.37 * (ac.af_S_h / ac.af_b_h ** 2) * (rho_0 * M_0 * c_0 * ac.af_S_h / (mu_0 * ac.af_b_h)) ** (-0.2)

        # Determine configuration constant and the sound power
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
        if ac.af_clean_h:
            K_h = 7.075e-6
        else:
            K_h = 4.464e-5
        Pi_star_h = K_h * M_0 ** 5 * delta_h_star * (ac.af_b_h / ac.af_b_w) ** 2

        # Determine the directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
        D_h = 4 * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

        # Determine spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
        S_h = freq * delta_h_star * ac.af_b_h / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        F_h = 0.485 * (10 * S_h) ** 4 * ((10 * S_h) ** 1.5 + 0.5) ** (-4)

        # Determine msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        r_s_star_af = settings.r_0 / ac.af_b_w
        msap_h = 1. / (4. * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (Pi_star_h * D_h * F_h)

        return msap_h

    @staticmethod
    def trailing_edge_vertical_tail(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, rho_0: np.float64, mu_0: np.float64, theta: np.float64, phi: np.float64, freq: np.ndarray) -> np.ndarray:
        """
        Compute vertical tail trailing edge mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param rho_0: ambient density [kg/m3]
        :type rho_0: np.float64
        :param mu_0: ambient dynamic viscosity [kg/m/s]
        :type mu_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_h
        :rtype: np.ndarray [settings.N_f]
        """

        ### ---------------- Vertical tail trailing-edge noise ----------------
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
        delta_v_star = 0.37 * (ac.af_S_v / ac.af_b_v ** 2) * (rho_0 * M_0 * c_0 * ac.af_S_v / (mu_0 * ac.af_b_v)) ** (-0.2)

        # Trailing edge noise of the vertical tail
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
        if ac.af_clean_v:
            K_v = 7.075e-6
        else:
            K_v = 4.464e-5
        Pi_star_v = K_v * M_0 ** 5 * delta_v_star * (ac.af_b_v / ac.af_b_w) ** 2

        # Determine directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
        D_v = 4 * np.sin(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

        # Determine spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
        S_v = freq * delta_v_star * ac.af_b_v / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        if ac.af_delta_wing:
            F_v = 0.613 * (10 * S_v) ** 4 * ((10 * S_v) ** 1.35 + 0.5) ** (-4)
        else:
            F_v = 0.485 * (10 * S_v) ** 4 * ((10 * S_v) ** 1.35 + 0.5) ** (-4)

        # Determine msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        r_s_star_af = settings.r_0 / ac.af_b_w
        msap_v = 1. / (4 * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (Pi_star_v * D_v * F_v)

        return msap_v

    @staticmethod
    def leading_edge_slat(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, rho_0: np.float64, mu_0: np.float64, theta: np.float64, phi: np.float64, freq: np.ndarray) -> np.ndarray:
        """
        Compute leading-edge slat mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param rho_0: ambient density [kg/m3]
        :type rho_0: np.float64
        :param mu_0: ambient dynamic viscosity [kg/m/s]
        :type mu_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_les
        :rtype: np.ndarray [settings.N_f]
        """

        ### ---------------- Slat noise ----------------
        delta_w_star = 0.37 * (ac.af_S_w / ac.af_b_w ** 2) * (rho_0 * M_0 * c_0 * ac.af_S_w / (mu_0 * ac.af_b_w)) ** (-0.2)

        # Noise power
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 4
        Pi_star_les1 = 4.464e-5 * M_0 ** 5 * delta_w_star  # Slat noise
        Pi_star_les2 = 4.464e-5 * M_0 ** 5 * delta_w_star  # Added trailing edge noise
        # Determine the directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
        D_les = 4 * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2
        # Determine spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-12-13
        S_les = freq * delta_w_star * ac.af_b_w / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        F_les1 = 0.613 * (10 * S_les) ** 4 * ((10. * S_les) ** 1.5 + 0.5) ** (-4)
        F_les2 = 0.613 * (2.19 * S_les) ** 4 * ((2.19 * S_les) ** 1.5 + 0.5) ** (-4)
        # Calculate msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        r_s_star_af = settings.r_0 / ac.af_b_w
        msap_les = 1 / (4 * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (Pi_star_les1 * D_les * F_les1 + Pi_star_les2 * D_les * F_les2)

        return msap_les

    @staticmethod
    def trailing_edge_flap(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, theta: np.float64, phi: np.float64, theta_flaps: np.float64, freq: np.ndarray) -> np.ndarray:
        """
        Compute trailing-edge flap mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param theta_flaps: flap deflection angle [deg]
        :type theta_flaps: np.float64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_tef
        :rtype: np.ndarray [settings.N_f]
        """
        ### ---------------- Flap noise ----------------
        # Calculate noise power
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 14-15
        if ac.af_s < 3:
            Pi_star_tef = 2.787e-4 * M_0 ** 6 * ac.af_S_f / ac.af_b_w ** 2 * np.sin(theta_flaps * np.pi / 180.) ** 2
        elif ac.af_s == 3:
            Pi_star_tef = 3.509e-4 * M_0 ** 6 * ac.af_S_f / ac.af_b_w ** 2 * np.sin(theta_flaps * np.pi / 180.) ** 2
        else:
            raise ValueError('Invalid number of flaps specified. No model available.')

        # Calculation of the directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 16
        D_tef = 3 * (np.sin(theta_flaps * np.pi / 180.) * np.cos(
            theta * np.pi / 180.) + np.cos(theta_flaps * np.pi / 180.) * np.sin(
            theta * np.pi / 180.) * np.cos(phi * np.pi / 180.)) ** 2

        # Strouhal number
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 19
        S_tef = freq * ac.af_S_f / (M_0 * ac.af_b_f * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        # Calculation of the spectral function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 17-18
        # if ac.af_s < 3:
        #     if S_tef < 2:
        #         F_tef = 0.0480 * S_tef
        #     elif 2 <= S_tef <= 20:
        #         F_tef = 0.1406 * S_tef ** (-0.55)
        #     else:
        #         F_tef = 216.49 * S_tef ** (-3)
        # elif ac.af_s == 3:
        #     if S_tef < 2:
        #         F_tef = 0.0257 * S_tef
        #     elif 2 <= S_tef <= 75:
        #         F_tef = 0.0536 * S_tef ** (-0.0625)
        #     else:
        #         F_tef = 17078 * S_tef ** (-3)
        if ac.af_s < 3:
            F_tef = 216.49 * S_tef ** (-3)
            F_tef[S_tef < 2] = (0.0480 * S_tef)[S_tef < 2]
            F_tef[(2 <= S_tef)*(S_tef <= 20)] = (0.1406 * S_tef ** (-0.55))[(2 <= S_tef)*(S_tef <= 20)]
        elif ac.af_s == 3:
            F_tef = 17078 * S_tef ** (-3)
            F_tef[S_tef < 2] = (0.0257 * S_tef)[S_tef < 2]
            F_tef[(2 <= S_tef)*(S_tef <= 75)] = (0.0536 * S_tef ** (-0.0625))[(2 <= S_tef)*(S_tef <= 75)]
        else:
            raise ValueError('Invalid number of flaps specified. No model available.')

        # Calculate msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        r_s_star_af = settings.r_0 / ac.af_b_w
        msap_tef = 1. / (4 * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (Pi_star_tef * D_tef * F_tef)

        return msap_tef

    @staticmethod
    def landing_gear(settings: Dict[str, Any], ac: Dict[str, Any], M_0: np.float64, c_0: np.float64, theta: np.float64, phi: np.float64, I_landing_gear: np.int64, freq: np.ndarray) -> np.ndarray:
        """
        Compute landing gear mean-square acoustic pressure (msap)

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param ac: aircraft parameters
        :type ac: Dict[str, Any]
        :param M_0: ambient Mach number [-]
        :type M_0: np.float64
        :param c_0: ambient speed of sound [m/s]
        :type c_0: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param phi: azimuthal directivity angle [deg]
        :type phi: np.float64
        :param I_landing_gear: landing gear deflection (0/1) [-]
        :type I_landing_gear: np.int64
        :param freq: 1/3rd octave frequency [Hz]
        :type freq: np.ndarray [settings.N_f]

        :return: msap_lg
        :rtype: np.ndarray [settings.N_f]
        """

        ### ---------------- Landing-gear noise ----------------
        if I_landing_gear == 1:
            # Calculate nose-gear noise
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
            S_ng = freq * ac.af_d_ng / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
            # Calculate noise power and spectral distribution function
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
            if ac.af_n_ng == 1 or ac.af_n_ng == 2:
                Pi_star_ng_w = 4.349e-4 * M_0 ** 6 * ac.af_n_ng * (ac.af_d_ng / ac.af_b_w) ** 2
                Pi_star_ng_s = 2.753e-4 * M_0 ** 6 * (ac.af_d_ng / ac.af_b_w) ** 2 * (ac.af_l_ng / ac.af_d_ng)
                F_ng_w = 13.59 * S_ng ** 2 * (12.5 + S_ng ** 2) ** (-2.25)
                F_ng_s = 5.32 * S_ng ** 2 * (30 + S_ng ** 8) ** (-1)
            elif ac.af_n_ng == 4:
                Pi_star_ng_w = 3.414 - 4 * M_0 ** 6 * ac.af_n_ng * (ac.af_d_ng / ac.af_b_w) ** 2
                Pi_star_ng_s = 2.753e-4 * M_0 ** 6 * (ac.af_d_ng / ac.af_b_w) ** 2 * (ac.af_l_ng / ac.af_d_ng)
                F_ng_w = 0.0577 * S_ng ** 2 * (1 + 0.25 * S_ng ** 2) ** (-1.5)
                F_ng_s = 1.28 * S_ng ** 3 * (1.06 + S_ng ** 2) ** (-3)
            else:
                raise ValueError('Invalid number of nose landing gear systems. Specify 1/2/4.')

            # Calculate main-gear noise
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
            S_mg = freq * ac.af_d_mg / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
            # Calculate noise power and spectral distribution function
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
            if ac.af_n_mg == 1 or ac.af_n_mg == 2:
                Pi_star_mg_w = 4.349e-4 * M_0 ** 6 * ac.af_n_mg * (ac.af_d_mg / ac.af_b_w) ** 2
                Pi_star_mg_s = 2.753e-4 * M_0 ** 6 * (ac.af_d_mg / ac.af_b_w) ** 2 * (ac.af_l_ng / ac.af_d_mg)
                F_mg_w = 13.59 * S_mg ** 2 * (12.5 + S_mg ** 2) ** (-2.25)
                F_mg_s = 5.32 * S_mg ** 2 * (30 + S_mg ** 8) ** (-1)
            elif ac.af_n_mg == 4:
                Pi_star_mg_w = 3.414e-4 * M_0 ** 6 * ac.af_n_mg * (ac.af_d_mg / ac.af_b_w) ** 2
                Pi_star_mg_s = 2.753e-4 * M_0 ** 6 * (ac.af_d_mg / ac.af_b_w) ** 2 * (ac.af_l_ng / ac.af_d_mg)
                F_mg_w = 0.0577 * S_mg ** 2 * (1 + 0.25 * S_mg ** 2) ** (-1.5)
                F_mg_s = 1.28 * S_mg ** 3 * (1.06 + S_mg ** 2) ** (-3)
            else:
                raise ValueError('Invalid number of main landing gear systems. Specify 1/2/4.')

            # Directivity function
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 23-24
            D_w = 1.5 * np.sin(theta * np.pi / 180.) ** 2
            D_s = 3 * np.sin(theta * np.pi / 180.) ** 2 * np.sin(phi * np.pi / 180.) ** 2
            # Calculate msap
            # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
            # If landing gear is down
            r_s_star_af = settings.r_0 / ac.af_b_w
            msap_lg = 1 / (4 * np.pi * r_s_star_af ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (
                                  ac.af_N_ng * (Pi_star_ng_w * F_ng_w * D_w + Pi_star_ng_s * F_ng_s * D_s) +
                                  ac.af_N_mg * (Pi_star_mg_w * F_mg_w * D_w + Pi_star_mg_s * F_mg_s * D_s))

        # If landing gear is up
        else:
            msap_lg = 0 * theta**0 * phi**0

        return msap_lg

    @staticmethod
    def airframe(source, theta, phi, inputs: openmdao.vectors.default_vector.DefaultVector) -> np.ndarray:
        """
        Compute airframe noise mean-square acoustic pressure (msap).

        :param source: pyNA component computing noise sources
		:type source: Source
        :param inputs: unscaled, dimensional input variables read via inputs[key]
        :type inputs: openmdao.vectors.default_vector.DefaultVector

        :return: msap_af
        :rtype: np.ndarray [n_t, settings.N_f]
        """
        # Load options
        settings = source.options['settings']
        data = source.options['data']
        ac = source.options['ac']
        n_t = source.options['n_t']

        # Extract inputs
        theta_flaps = inputs['theta_flaps']
        I_landing_gear = np.round(inputs['I_landing_gear'])
        M_0 = inputs['M_0']
        c_0 = inputs['c_0']
        rho_0 = inputs['rho_0']
        mu_0 = inputs['mu_0']

        # Initialize solution
        msap_af = np.zeros((n_t, settings.N_f))

        # Compute airframe
        for i in np.arange(n_t):

            # Calculate msap when the aircraft is not at standstill
            if M_0[i] != 0:

                # Apply HSR-era airframe calibration levels
                if settings.hsr_calibration:
                    # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
                    supp = data.supp_af_f(theta[i], data.f).reshape(settings.N_f, )
                else:
                    supp = np.ones(settings.N_f)

                # Add airframe noise components
                msap_j = np.zeros(settings.N_f)
                if 'wing' in ac.comp_lst:
                    msap_w = Airframe.trailing_edge_wing(settings, ac, M_0[i], c_0[i], rho_0[i], mu_0[i], theta[i], phi[i], data.f)
                    msap_j = msap_j + msap_w * supp
                if 'tail_v' in ac.comp_lst:
                    msap_v = Airframe.trailing_edge_vertical_tail(settings, ac, M_0[i], c_0[i], rho_0[i], mu_0[i], theta[i], phi[i], data.f)
                    msap_j = msap_j + msap_v * supp
                if 'tail_h' in ac.comp_lst:
                    msap_h = Airframe.trailing_edge_horizontal_tail(settings, ac, M_0[i], c_0[i], rho_0[i], mu_0[i], theta[i], phi[i], data.f)
                    msap_j = msap_j + msap_h * supp
                if 'les' in ac.comp_lst:
                    msap_les = Airframe.leading_edge_slat(settings, ac, M_0[i], c_0[i], rho_0[i], mu_0[i], theta[i], phi[i], data.f)
                    msap_j = msap_j + msap_les * supp
                if 'tef' in ac.comp_lst:
                    msap_tef = Airframe.trailing_edge_flap(settings, ac, M_0[i], c_0[i], theta[i], phi[i], theta_flaps[i], data.f)
                    msap_j = msap_j + msap_tef * supp
                if 'lg' in ac.comp_lst:
                    msap_lg = Airframe.landing_gear(settings, ac, M_0[i], c_0[i], theta[i], phi[i], I_landing_gear[i], data.f)
                    msap_j = msap_j + msap_lg * supp

            else:
                msap_j = 1e-99 * (np.ones(settings.N_f) * theta_flaps[i]) ** 0

            # Normalize msap by reference pressure
            msap_af[i, :] = msap_j / settings.p_ref ** 2

        return msap_af

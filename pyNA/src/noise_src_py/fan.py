import pdb
import openmdao
import openmdao.api as om
import numpy as np

from typing import Dict, Any, Tuple


class Fan:

    @staticmethod
    def inlet_broadband(settings:Dict[str, Any], theta: np.float64, M_tip: np.float64, tsqem: np.float64, M_d_fan: np.float64, RSS_fan: np.float64) -> np.float64:
        """
        Compute the broadband component of the fan inlet mean-square acoustic pressure (msap).

        :param settings: pyna settings.
        :type settings: Dict[str, Any]
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param tsqem: broadband temperature-flow power base term [-]
        :type tsqem: np.float64
        :param M_d_fan: fan rotor relative tip Mach number at design [-]
        :type M_d_fan: np.float64
        :param RSS_fan: fan rotor-stator spacing [%]
        :type RSS_fan: np.float64

        :return: bblv_I
        :rtype: np.float64
        """

        # Fan inlet broadband noise component:
        if settings.fan_BB_method == 'original':
            # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A):
            if M_d_fan <= 1:
                if M_tip <= 0.9:
                    F1IB = 58.5
                else:
                    F1IB = 58.5 - 20 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    F1IB = 58.5 + 20 * np.log10(M_d_fan)
                else:
                    F1IB = 58.5 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 0.9)

            # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
            if not settings.fan_id:
                F2IB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2IB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2IB = -5 * np.log10(100 / 300)  # This is set to a constant 2.3856
        elif settings.fan_BB_method == 'allied_signal':
            # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by AlliedSignal):
            if M_d_fan <= 1:
                if M_tip <= 0.9:
                    F1IB = 55.5
                else:
                    F1IB = 55.5 - 20 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    F1IB = 55.5 + 20 * np.log10(M_d_fan)
                else:
                    F1IB = 55.5 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 0.9)

            # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
            if not settings.fan_id:
                F2IB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2IB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2IB = -5 * np.log10(100. / 300)  # This is set to a constant 2.3856
        elif settings.fan_BB_method == 'geae':
            # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by GE):
            if M_d_fan <= 1:
                if M_tip <= 0.9:
                    F1IB = 58.5
                else:
                    F1IB = 58.5 - 50 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    F1IB = 58.5 + 20 * np.log10(M_d_fan)
                else:
                    F1IB = 58.5 + 20 * np.log10(M_d_fan) - 50 * np.log10(M_tip / 0.9)

            # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
            F2IB = 0
        elif settings.fan_BB_method == 'kresja':
            # Tip Mach-dependent term (F1, of Eqn 4 in report, Figure 4A, modified by Krejsa):
            if M_d_fan <= 1:
                if M_tip < 0.72:
                    F1IB = 34 + 20 * np.log10(1. / 1.245)
                else:
                    F1IB = 34 - 43 * (M_tip - 0.72) + 20 * np.log10(1. / 1.245)
            else:
                if M_tip < 0.72:
                    F1IB = 34 + 20 * np.log10(M_d_fan / 1.245)
                else:
                    F1IB = 34 - 43 * (M_tip - 0.72) + 20 * np.log10(M_d_fan/ 1.245)

            # Rotor-stator spacing correction term (F2, of Eqn 4, Figure 6B):
            if not settings.fan_id:
                F2IB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2IB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2IB = -5 * np.log10(100 / 300)  # This is set to a constant 2.3856
        else:
            raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

        # Theta correction term (F3 of Eqn 4, Figure 7A):
        if settings.fan_BB_method == 'kresja':
            # Krejsa method:
            THET7A = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
            FIG7A = np.array([-0.5, -1, -1.25, -1.41, -1.4, -2.2, -4.5, -8.5, -13, -18.5, -24, -30, -36, -42, -48, -54, -60,-66, -73, -66])
            F3IB = np.interp(theta, THET7A, FIG7A)
        else:
            # All other methods:
            THET7A = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 180, 250])
            FIG7A = np.array([-2, -1, 0, 0, 0, -2, -4.5, -7.5, -11, -15, -19.5, -25, -63.5, -25])
            F3IB = np.interp(theta, THET7A, FIG7A)

        # Component value:
        bblv_I = tsqem + F1IB + F2IB + F3IB

        return bblv_I

    @staticmethod
    def discharge_broadband(settings:Dict[str, Any], theta: np.float64, M_tip: np.float64, tsqem: np.float64, M_d_fan: np.float64, RSS_fan: np.float64) -> np.float64:
        """
        Compute the broadband component of the fan discharge mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param tsqem: broadband temperature-flow power base term [-]
        :type tsqem: np.float64
        :param M_d_fan: fan rotor relative tip Mach number at design [-]
        :type M_d_fan: np.float64
        :param RSS_fan: fan rotor-stator spacing [%]
        :type RSS_fan: np.float64

        :return: bblv_D
        :rtype: np.float64
        """
        # Fan discharge broadband noise component
        if settings.fan_BB_method == 'original':
            # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1DB = 60
                else:
                    F1DB = 60 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1DB = 60 + 20 * np.log10(M_d_fan)
                else:
                    F1DB = 60 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 1)

            # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
            if not settings.fan_id:
                F2DB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2DB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2DB = -5 * np.log10(100 / 300)  # This is set to a constant 2.3856
        elif settings.fan_BB_method == 'allied_signal':
            # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by AlliedSignal):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1DB = 58
                else:
                    F1DB = 58 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1DB = 58 + 20 * np.log10(M_d_fan)
                else:
                    F1DB = 58 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 1)

            # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by AlliedSignal):
            if not settings.fan_id:
                F2DB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2DB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2DB = -5 * np.log10(100 / 300)  # This is set to a constant 2.3856
        elif settings.fan_BB_method == 'geae':
            # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by GE):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1DB = 63
                else:
                    F1DB = 63 - 30 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1DB = 63 + 20 * np.log10(M_d_fan)
                else:
                    F1DB = 63 + 20 * np.log10(M_d_fan) - 30 * np.log10(M_tip / 1)

            # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by GE):
            F2DB = -5 * np.log10(RSS_fan / 300)
        elif settings.fan_BB_method == 'kresja':
            # Tip Mach-dependent term (F1, of Eqn 10 in report, Figure 4B, modified by Krejsa):
            if M_d_fan <= 1:
                # If M_tip < 0.65 Then
                #   F1DBBkrejsa = 34 + 20 * np.log10(1 / 1.245)
                # Else
                F1DB = 34 - 17 * (M_tip - 0.65) + 20 * np.log10(1 / 1.245)
            else:
                # If M_tip < 0.65 Then
                #   F1DBBkrejsa = 34 + 20 * np.log10(M_d_fan / 1.245)
                # Else
                F1DB = 34 - 17 * (M_tip - 0.65) + 20 * np.log10(M_d_fan / 1.245)

            # Rotor-stator spacing correction term (F2, of Eqn 10, Figure 6B):
            if not settings.fan_id:
                F2DB = -5 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan <= 100:
                    F2DB = -5 * np.log10(RSS_fan / 300)
                else:
                    F2DB = -5 * np.log10(100 / 300)  # This is set to a constant 2.3856
        else:
            raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

        # Theta correction term (F3 of Eqn 10, Figure 7B):
        if settings.fan_BB_method == 'allied_signal':
            THET7B = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
            FIG7B = np.array([0, -29.5, -26, -22.5, -19, -15.5, -12, -8.5, -5, -3.5, -2.5, -2, -1.3, 0, -3, -7, -11, -15, -20])
            F3DB = np.interp(theta, THET7B, FIG7B)
        elif settings.fan_BB_method == 'kresja':
            THET7B = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
            FIG7B = np.array([-30, -25, -20.8, -19.5, -18.4, -16.7, -14.5, -12, -9.6, -6.9, -4.5, -1.8, -0.3, 0.5, 0.7, -1.9,-4.5, -9, -15, -9])
            F3DB = np.interp(theta, THET7B, FIG7B)
        else:  # For original and GE large fan methods:
            THET7B = np.array([0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
            FIG7B = np.array([-41.6, -15.8, -11.5, -8, -5, -2.7, -1.2, -0.3, 0, -2, -6, -10, -15, -20, -15])
            F3DB = np.interp(theta, THET7B, FIG7B)

        # Added noise factor if there are inlet guide vanes present:
        if settings.fan_igv:
            CDB = 3
        else:
            CDB = 0

        # Component value:
        bblv_D = tsqem + F1DB + F2DB + F3DB + CDB

        return bblv_D

    @staticmethod
    def inlet_tones(settings:Dict[str, Any], theta: np.float64, M_tip: np.float64, tsqem: np.float64, M_d_fan: np.float64, RSS_fan: np.float64) -> np.float64:
        """
        Compute the tone component of the fan inlet mean-square acoustic pressure (msap)

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param tsqem: tone temperature-flow power base term [-]
        :type tsqem: np.float64
        :param M_d_fan: fan rotor relative tip Mach number at design [-]
        :type M_d_fan: np.float64
        :param RSS_fan: fan rotor-stator spacing [%]
        :type M_d_fan: np.float64

        :return: tonlv_I
        :rtype: np.float64
        """

        # Fan inlet discrete tone noise component:

        if settings.fan_RS_method == 'original':
            # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A):
            if M_d_fan <= 1:
                if M_tip <= 0.72:
                    F1TI = 60.5
                else:
                    F1TIA = 60.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 59.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB
            else:
                if M_tip <= 0.72:
                    F1TI = 60.5 + 20 * np.log10(M_d_fan)
                else:
                    F1TIA = 60.5 + 20 * np.log10(M_d_fan) + 50 * np.log10(M_tip / 0.72)
                    # Note the 1975 version of NASA TMX-71763 writes it this way:
                    # F1TIB = 59.5 + 20 * np.log10(M_d_fan) + 80 * np.log10(M_d_fan / M_tip)
                    # But the 1979 version of NASA TMX-71763 writes it this way:
                    F1TIB = 59.5 + 80 * np.log10(M_d_fan / M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB

            # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
            if not settings.fan_id:
                F2TI = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TI = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TI = -10 * np.log10(100 / 300)  # This is set to a constant 4.7712
        elif settings.fan_RS_method == 'allied_signal':
            # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by AlliedSignal):
            if M_d_fan <= 1:
                if M_tip <= 0.72:
                    F1TI = 54.5
                else:
                    F1TIA = 54.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 53.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB
            else:
                if M_tip <= 0.72:
                    F1TI = 54.5 + 20 * np.log10(M_d_fan)
                else:
                    F1TIA = 54.5 + 20 * np.log10(M_d_fan) + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 53.5 + 80 * np.log10(M_d_fan / M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB

            # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified by AlliedSignal):
            if not settings.fan_id:
                F2TI = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TI = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TI = -10 * np.log10(100./ 300)  # This is set to a constant 4.7712
        elif settings.fan_RS_method == 'geae':
            # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by GE):
            if M_d_fan <= 1:
                if M_tip <= 0.72:
                    F1TI = 60.5
                else:
                    F1TIA = 60.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 64.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB
            else:
                if M_tip <= 0.72:
                    F1TI = 60.5 + 20 * np.log10(M_d_fan)
                else:
                    F1TIA = 60.5 + 20 * np.log10(M_d_fan) + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 64.5 + 80 * np.log10(M_d_fan) - 80 * np.log10(M_tip)
                    if F1TIA < F1TIB:
                        F1TI = F1TIA
                    else:
                        F1TI = F1TIB

            # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified to zero by GE):
            F2TI = 0
        elif settings.fan_RS_method == 'kresja':
            # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by Krejsa):
            if M_d_fan <= 1:
                F1TI = 42 - 20 * M_tip + 20 * np.log10(1. / 1.245)
            else:
                F1TI = 42 - 20 * M_tip + 20 * np.log10(M_d_fan / 1.245)

            # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
            if not settings.fan_id:
                F2TI = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TI = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TI = -10 * np.log10(100 / 300)  # This is set to a constant 4.7712
        else:
            raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

        # Theta correction term (F3 of Eqn 6, Figure 13A):
        if settings.fan_RS_method == 'allied_signal':
            THT13A = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
            FIG13A = np.array([-3, -1.5, -1.5, -1.5, -1.5, -2, -3, -4, -6, -9, -12.5, -16, -19.5, -23, -26.5, -30, -33.5, -37,-40.5])
            F3TI = np.interp(theta, THT13A, FIG13A)

        elif settings.fan_RS_method == 'kresja':
            THT13A = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
            FIG13A = np.array([-3, -1.5, 0, 0, 0, -1.2, -3.5, -6.8, -10.5, -15.5, -19, -25, -32, -40, -49, -59, -70, -80, -90])
            F3TI = np.interp(theta, THT13A, FIG13A)
        else:  # For original and GE large fan methods:
            THT13A = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 180, 260])
            FIG13A = np.array([-3, -1.5, 0, 0, 0, -1.2, -3.5, -6.8, -10.5, -14.5, -19, -55, -19])
            F3TI = np.interp(theta, THT13A, FIG13A)

        # Component value:
        tonlv_I = tsqem + F1TI + F2TI + F3TI

        return tonlv_I

    @staticmethod
    def discharge_tones(settings:Dict[str, Any], theta: np.float64, M_tip: np.float64, tsqem: np.float64, M_d_fan: np.float64, RSS_fan: np.float64) -> np.float64:
        """
        Compute the tone component of the fan discharge mean-square acoustic pressure (msap)

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param tsqem: broadband temperature-flow power base term [-]
        :type tsqem: np.float64
        :param M_d_fan: fan rotor relative tip Mach number at design [-]
        :type M_d_fan: np.float64
        :param RSS_fan: fan rotor-stator spacing [%]
        :type M_d_fan: np.float64

        :return: tonlv_X
        :rtype: np.float64
        """

        # Fan discharge discrete tone noise component:

        if settings.fan_RS_method == 'original':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1TD = 63
                else:
                    F1TD = 63 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1TD = 63 + 20 * np.log10(M_d_fan)
                else:
                    F1TD = 63 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 1)

            # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12):
            if not settings.fan_id:
                F2TD = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TD = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TD = -10 * np.log10(100 / 300)  # This is set to a constant 4.7712
        elif settings.fan_RS_method == 'allied_signal':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by AlliedSignal):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1TD = 59
                else:
                    F1TD = 59 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1TD = 59 + 20 * np.log10(M_d_fan)
                else:
                    F1TD = 59 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 1)

            # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by AlliedSignal):
            if not settings.fan_id:
                F2TD = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TD = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TD = -10 * np.log10(100 / 300)  # This is set to a constant 4.7712
        elif settings.fan_RS_method == 'geae':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by GE):
            if M_d_fan <= 1:
                if M_tip <= 1:
                    F1TD = 63
                else:
                    F1TD = 63 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    F1TD = 63 + 20 * np.log10(M_d_fan)
                else:
                    F1TD = 63 + 20 * np.log10(M_d_fan) - 20 * np.log10(M_tip / 1)

            # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by GE):
            F2TD = -10 * np.log10(RSS_fan / 300)
        elif settings.fan_RS_method == 'kresja':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by Krejsa):
            if M_d_fan <= 1:
                F1TD = 46 - 20 * M_tip + 20 * np.log10(1 / 1.245)
            else:
                F1TD = 46 - 20 * M_tip + 20 * np.log10(M_d_fan / 1.245)

            # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by Krejsa):
            if not settings.fan_id:
                F2TD = -10 * np.log10(RSS_fan / 300)  # If no distortion
            else:
                if RSS_fan < 100:
                    F2TD = -10 * np.log10(RSS_fan / 300)
                else:
                    F2TD = -10 * np.log10(100 / 300)  # This is set to a constant 4.7712
        else:
            raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

        # Theta correction term (F3 of Eqn 6, Figure 13B):
        if settings.fan_RS_method == 'allied_signal':
            THT13B = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
            FIG13B = np.array([-34, -30, -26, -22, -18, -14, -10.5, -6.5, -4, -1, 0, 0, 0, 0, -1, -3.5, -7, -11, -16])
            F3TD = np.interp(theta, THT13B, FIG13B)
        elif settings.fan_RS_method == 'kresja':
            THT13B = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
            FIG13B = np.array([-50, -41, -33, -26, -20.6, -17.9, -14.7, -11.2, -9.3, -7.1, -4.7, -2, 0, 0.8, 1, -1.6, -4.2, -9,-15])
            F3TD = np.interp(theta, THT13B, FIG13B)
        elif settings.fan_RS_method in ['original', 'geae']:
            THT13B = np.array([0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
            FIG13B = np.array([-39, -15, -11, -8, -5, -3, -1, 0, 0, -2, -5.5, -9, -13, -18, -13])
            F3TD = np.interp(theta, THT13B, FIG13B)
        else:
            raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

        # Added noise factor if there are inlet guide vanes:
        if settings.fan_igv:
            CDT = 6
        else:
            CDT = 0

        # Component value:
        tonlv_X = tsqem + F1TD + F2TD + F3TD + CDT

        return tonlv_X

    @staticmethod
    def combination_tones(settings:Dict[str, Any], freq: np.float64, theta: np.float64, M_tip: np.float64, bpf: np.float64, tsqem: np.float64) -> np.ndarray:
        """
        Compute the combination tone component of the fan mean-square acoustic pressure (msap).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param freq: 1/3rd octave frequency bands [Hz]
        :type freq: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param bpf: blade pass frequency
        :type bpf: np.float64
        :param tsqem: tone temperature-flow power base term [-]
        :type tsqem: np.float64

        :return: dcp
        :rtype: np.float64
        """
        # Combination tone (multiple pure tone or buzzsaw) calculations:
        # Note the original Heidmann reference states that MPTs should be computed if
        # the tangential tip speed is supersonic, but the ANOPP implementation states MPTs
        # should be computed if the relative tip speed is supersonic.  The ANOPP implementation
        # is used here, i.e., if M_tip >= 1.0, MPTs are computed.

        # Initialize solution matrices
        dcp = np.zeros(settings.N_f)

        if M_tip >= 1:
            if settings.fan_RS_method == 'original':
                # Theta correction term (F2 of Eqn 8, Figure 16), original method:
                THT16 = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 180, 270])
                FIG16 = np.array([-9.5, -8.5, -7, -5, -2, 0, 0, -3.5, -7.5, -9, -13.5, -9])
                F2CT = np.interp(theta, THT16, FIG16)

                # Spectrum slopes, original method:
                SL3 = np.array([0, -30, -50, -30])
                SL4 = np.array([0, 30, 50, 50])
                YINT3 = np.array([0, -9.0309, -30.103, -27.0927])
                YINT4 = np.array([0, 9.0309, 30.103, 45.1545])
            elif settings.fan_RS_method == 'allied_signal':
                # Theta correction term (F2 of Eqn 8, Figure 16), small fan method:
                THT16 = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 270])
                FIG16 = np.array([-5.5, -4.5, -3, -1.5, 0, 0, 0, 0, -2.5, -5, -6, -6.9, -7.9, -8.8, -9.8, -10.7, -11.7, -12.6,-13.6, -6])
                F2CT = np.interp(theta, THT16, FIG16)
                # Spectrum slopes, small fan method:
                SL3 = np.array([0, -15, -50, -30])
                SL4 = np.array([0, 30, 50, 50])
                YINT3 = np.array([0, -4.51545, -30.103, -27.0927])
                YINT4 = np.array([0, 9.0309, 30.103, 45.1545])
            elif settings.fan_RS_method == 'geae':
                # Theta correction term (F2 of Eqn 8, Figure 16), large fan method:
                THT16 = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 180, 270])
                FIG16 = np.array([-9.5, -8.5, -7, -5, -2, 0, 0, -3.5, -7.5, -9, -13.5, -9])
                F2CT = np.interp(theta, THT16, FIG16)
                # Spectrum slopes, GE large fan method:
                SL3 = np.array([0, -30, -50, -30])
                SL4 = np.array([0, 30, 50, 50])
                YINT3 = np.array([0, -9.0309, -30.103, -27.0927])
                YINT4 = np.array([0, 9.0309, 30.103, 45.1545])
            elif settings.fan_RS_method == 'kresja':
                # Theta correction term (F2 of Eqn 8, Figure 16), Krejsa method:
                THT16 = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
                FIG16 = np.array([-28, -23, -18, -13, -8, -3, 0, -1.3, -2.6, -3.9, -5.2, -6.5, -7.9, -9.4, -11, -12.7, -14.5,-16.4, -18.4])
                F2CT = np.interp(theta, THT16, FIG16)
                # Spectrum slopes, Krejsa method:
                SL3 = np.array([0, -20, -30, -20])
                SL4 = np.array([0, 20, 30, 30])
                YINT3 = np.array([0, -6.0206, -18.0618, -18.0618])
                YINT4 = np.array([0, 6.0206, 18.0618, 27.0927])
            else:
                raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

            # Noise adjustment (reduction) if there are inlet guide vanes, for all methods:
            if settings.fan_igv:
                CCT = -5
            else:
                CCT = 0

            # Loop through the three sub-bpf terms:
            # K = 1; 1/2 bpf term
            # K = 2; 1/4 bpf term
            # K = 3; 1/8 bpf term
            for K in np.arange(1, 4):
                # Tip Mach-dependent term (F1 of Eqn 8 in Heidmann report, Figure 15A):
                if settings.fan_RS_method == 'original':
                    # Original tip Mach number-dependent term of multiple pure tone noise
                    FIG15MT = np.array([0, 1.14, 1.25, 1.61])
                    SL1 = np.array([0, 785.68, 391.81, 199.2])
                    SL2 = np.array([0, -49.62, -50.06, -49.89])
                    YINT1 = np.array([0, 30, 30, 30])
                    YINT2 = np.array([0, 79.44, 83.57, 81.52])

                    if M_tip < FIG15MT[K]:
                        F1CT = SL1[K] * np.log10(M_tip) + YINT1[K]
                    else:
                        F1CT = SL2[K] * np.log10(M_tip) + YINT2[K]
                elif settings.fan_RS_method == 'allied_signal':
                    # Revised by AlliedSignal:  tip Mach number-dependent term of multiple pure tone noise.
                    # Note that a 20log10 independent variable distribution is specified.
                    FIG15MT = np.array([0, 1.135, 1.135, 1.413])
                    SL1 = np.array([0, 5.9036, 5.54769, 2.43439])
                    SL2 = np.array([0, -0.632839, -0.632839, -1.030931])
                    YINT1 = np.array([0, 50, 50, 40])
                    YINT2 = np.array([0, 57.1896, 56.7981, 50.4058])

                    if M_tip < FIG15MT[K]:
                        F1CT = SL1[K] * 20 * np.log10(M_tip) + YINT1[K]
                    else:
                        F1CT = SL2[K] * 20 * np.log10(M_tip) + YINT2[K]
                elif settings.fan_RS_method == 'geae':
                    # Revised by GE: tip Mach number-dependent term of multiple pure tone noise
                    FIG15MT = np.array([0, 1.14, 1.25, 1.61])
                    SL1 = np.array([0, 746.8608, 398.3077, 118.9406])
                    SL2 = np.array([0, -278.96, -284.64, -43.52])
                    YINT1 = np.array([0, 30, 30, 36])
                    YINT2 = np.array([0, 88.37, 96.18, 69.6])

                    if M_tip < FIG15MT[K]:
                        F1CT = SL1[K] * np.log10(M_tip) + YINT1[K]
                    else:
                        F1CT = SL2[K] * np.log10(M_tip) + YINT2[K]
                elif settings.fan_RS_method == 'kresja':
                    FIG15MT = np.array([0, 1.146, 1.322, 1.61])
                    SL1 = np.array([0, 785.68, 391.81, 199.2])
                    SL2 = np.array([0, -49.62, -50.06, -49.89])
                    YINT1 = np.array([0, -18, -15, -12])
                    YINT2 = np.array([0, 31.44, 38.57, 39.52])

                    if M_tip < FIG15MT[K]:
                        F1CT = SL1[K] * np.log10(M_tip) + YINT1[K]
                    else:
                        F1CT = SL2[K] * np.log10(M_tip) + YINT2[K]
                else:
                    raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

                CTLC = tsqem + F1CT + F2CT + CCT
                # Frequency-dependent term (F3 of Eqn 9, Figure 14):
                FK = 2 ** K

                # Cycle through frequencies and make assignments:
                for j in np.arange(settings.N_f):
                    FQFB = freq[j] / bpf

                    if FQFB <= 1 / FK:
                        # For frequencies less than the subharmonic:
                        F3CT = SL4[K] * np.log10(FQFB) + YINT4[K]
                    else:
                        # For frequencies greater than the subharmonic:
                        F3CT = SL3[K] * np.log10(FQFB) + YINT3[K]

                    # Be sure to add the three sub-bpf components together at each frequency:
                    dcp[j] = dcp[j] + 10 ** (0.1 * (CTLC + F3CT))

        return dcp

    @staticmethod
    def calculate_cutoff(M_tip_tan: np.float64, B_fan: np.int64, V_fan: np.int64) -> np.float64:
        """
        Compute if the fan is in cut-off condition (0/1).

        :param M_tip_tan: tangential (i.e., radius*omega) tip Mach number [-]
        :type M_tip_tan: np.float64
        :param B_fan: fan blade number [-]
        :type B_fan: np.int64
        :param V_fan: fan vane number [-]
        :type V_fan: np.int64

        :return: i_cut
        :rtype: np.int64
        """

        # Vane/blade ratio parameter:
        vane_blade_ratio = 1 - V_fan / B_fan
        if vane_blade_ratio == 0:
            vane_blade_ratio = 1e-6

        # Fundamental tone cutoff parameter:
        # Source: Zorumski report 1982 part 2. Chapter 8.1 Eq. 8
        delta_cutoff = abs(M_tip_tan / vane_blade_ratio)
        # if the cutoff parameter is less than 1.05 and the tip Mach is less than unity, the fan is cut off
        # and fan noise does not propagate (i.e., the tones are reduced in magnitude):
        if delta_cutoff < 1.05:
            # Fan cut-off
            if M_tip_tan < 1:
                i_cut = delta_cutoff ** 0
            # No cutoff: supersonic tip mach number
            else:
                i_cut = delta_cutoff ** 0 - 1
        else:
            # No cutoff: poor choice of blades and vanes
            if M_tip_tan < 1:
                i_cut = delta_cutoff ** 0 - 1
            # No cutoff: supersonic tip mach number
            else:
                i_cut = delta_cutoff ** 0 - 1

        return i_cut

    @staticmethod
    def calculate_harmonics(settings:Dict[str, Any], freq: np.float64, theta: np.float64, tonlv_I: np.float64, tonlv_X: np.float64, i_cut: np.int64, M_tip: np.float64, bpf: np.float64, comp: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fan tone harmonics for inlet (dp) and discharge (dpx).

        :param settings: pyna settings
        :type settings: Dict[str, Any]
        :param freq: 1/3rd octave frequency bands [Hz]
        :type freq: np.float64
        :param theta: polar directivity angle [deg]
        :type theta: np.float64
        :param tonlv_I: inlet tone level [-]
        :type tonlv_I: np.float64
        :param tonlv_X: discharge tone level [-]
        :type tonlv_X: np.float64
        :param i_cut: cut-off parameter (0/1)
        :type i_cut: np.int64
        :param M_tip: relative (i.e., helical) tip Mach number [-]
        :type M_tip: np.float64
        :param bpf: blade pass frequency
        :param comp: fan component (fan_inlet / fan_discharge)
        :type comp: str

        :return: dp, dpx
        :rtype: np.ndarray [settings.N_f], np.ndarray [settings.N_f]

        """

        # Assign discrete interaction tones at bpf and harmonics to proper bins (see figures 8 and 9):
        # Initialize solution matrices
        dp = np.zeros(settings.N_f)
        dpx = np.zeros(settings.N_f)

        nfi = 1
        for ih in np.arange(1, settings.n_harmonics + 1):

            # Determine the tone fall-off rates per harmonic (harm_i and harm_x):
            if settings.fan_RS_method == 'original':
                if not settings.fan_igv:
                    # For fans without inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
                else:
                    # For fans with inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
            elif settings.fan_RS_method == 'allied_signal':
                if not settings.fan_igv:
                    # For fans without inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        elif ih == 2:
                            # For cut-on fans, second harmonic:
                            harm_i = 9.2
                            harm_x = 9.2
                        else:
                            # For cut-on fans, upper harmonics:
                            harm_i = 3 * ih + 1.8
                            harm_x = 3 * ih + 1.8
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        elif ih == 2:
                            # For cut-off fans, second harmonic:
                            harm_i = 9.2
                            harm_x = 9.2
                        else:
                            # For cut-off fans, upper harmonics:
                            harm_i = 3 * ih + 1.8
                            harm_x = 3 * ih + 1.8
                    else:
                        raise ValueError('Cut-off value out of bounds.')
                else:
                    # For fans with inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
            elif settings.fan_RS_method == 'geae':
                if not settings.fan_igv:
                    # For fans without inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            if M_tip < 1.15:
                                harm_i = 6 * (ih - 1)
                                harm_x = 3 * (ih - 1)
                            else:
                                harm_i = 9 * (ih - 1)
                                harm_x = 3 * (ih - 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            if M_tip < 1.15:
                                harm_i = 6 * (ih - 1)
                                harm_x = 3 * (ih - 1)
                            else:
                                harm_i = 9 * (ih - 1)
                                harm_x = 3 * (ih - 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
                else:
                    # For fans with inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
            elif settings.fan_RS_method == 'kresja':
                if not settings.fan_igv:
                    # For fans without inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
                else:
                    # For fans with inlet guide vanes:
                    if i_cut == 0:
                        # For cut-on fans, fundamental:
                        if ih == 1:
                            harm_i = 0
                            harm_x = 0
                        else:
                            # For cut-on fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    elif i_cut == 1:
                        # For cut-off fans, fundamental:
                        if ih == 1:
                            harm_i = 8
                            harm_x = 8
                        else:
                            # For cut-off fans, harmonics:
                            harm_i = 3 * (ih + 1)
                            harm_x = 3 * (ih + 1)
                    else:
                        raise ValueError('Cut-off value out of bounds.')
            else:
                raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

            # Calculate TCS and distor
            if settings.fan_id and settings.fan_RS_method != 'geae':
                # Assign the increment to the fundamental tone along with a 10 dB per harmonic order fall-off
                # for cases with inlet flow distortion (see figure 9):
                distor = 10 ** (0.1 * tonlv_I - ih + 1)
                TCS = 0
            elif settings.fan_id and settings.fan_RS_method == 'geae':
                # Compute suppression factors for GE#s "Flight cleanup Turbulent Control Structure."
                # Approach or takeoff values to be applied to inlet discrete interaction tones
                # at bpf and 2bpf.  Accounts for observed in-flight tendencies.
                TCSTHA = np.array([0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
                TCSAT1 = np.array([0, 4.8, 4.8, 5.5, 5.5, 5.3, 5.3, 5.1, 4.4, 3.9, 2.6, 2.3, 1.8, 2.1, 1.7, 1.7, 2.6, 3.5, 3.5,3.5])
                TCSAT2 = np.array([0, 5.8, 5.8, 3.8, 5.3, 6.4, 3.5, 3, 2.1, 2.1, 1.1, 1.4, 0.9, 0.7, 0.7, 0.4, 0.6, 0.8, 0.8,0.8])
                TCSAA1 = np.array([0, 5.6, 5.6, 5.8, 4.7, 4.6, 4.9, 5.1, 2.9, 3.2, 1.6, 1.6, 1.8, 2.1, 2.4, 2.2, 2, 2.8, 2.8,2.8])
                TCSAA2 = np.array([0, 5.4, 5.4, 4.3, 3.4, 4.1, 2, 2.9, 1.6, 1.3, 1.5, 1.1, 1.4, 1.5, 1, 1.8, 1.6, 1.6, 1.6, 1.6])

                if settings.ge_flight_cleanup == 'takeoff':
                    # Apply takeoff values:
                    if ih == 1:
                        TCS = np.interp(theta, TCSTHA, TCSAT1)
                    elif ih == 2:
                        TCS = np.interp(theta, TCSTHA, TCSAT2)
                    else:
                        TCS = 0
                elif settings.ge_flight_cleanup == 'approach':
                    # Apply approach values:
                    if ih == 1:
                        TCS = np.interp(theta, TCSTHA, TCSAA1)
                    elif ih == 2:
                        TCS = np.interp(theta, TCSTHA, TCSAA2)
                    else:
                        TCS = 0
                elif settings.ge_flight_cleanup == 'none':
                    # Apply zero values (i.e., fan inlet flow is distorted):
                    TCS = 0
                else:
                    raise ValueError('Invalid ge_flight_cleanup method specified. Specify: takeoff / approach / none')

                # Inlet distortion effects are always included in basic inlet tone model of the GE method.
                # Flight cleanup levels are then subtracted from the inlet tones if the flow is not distorted.
                # The flight cleanup levels are set to zero if the flow is distorted.
                # Use the same increment as the original method and the same 10 dB per harmonic fall-off rate.
                distor = 10 ** (0.1 * (tonlv_I - TCS) - ih + 1)
            else:
                distor = 0
                TCS = 0

            # Calculate tone power
            if comp == 'fan_inlet':# or comp == 'inlet RS':
                tonpwr_i = 10 ** (0.1 * (tonlv_I - harm_i - TCS)) + distor
            else:
                tonpwr_i = 0

            if comp == 'fan_discharge':# or comp == 'discharge RS' or comp == 'total':
                tonpwr_x = 10 ** (0.1 * (tonlv_X - harm_x))
            else:
                tonpwr_x = 0

            # Compute filter bandwidths:
            filbw = 1  # Fraction of filter bandwidth with gain of unity (default to unity)
            F1 = 0.78250188 + 0.10874906 * filbw
            F2 = 1 - 0.10874906 * filbw
            F3 = 1 + 0.12201845 * filbw
            F4 = 1.2440369 - 0.12201845 * filbw

            # Cycle through frequencies and assign tones to 1/3rd octave bins:
            for l in np.arange(nfi - 1, settings.N_f):
                Frat = bpf * ih / freq[l]
                FR = 1
                if Frat < F1:
                    break
                elif Frat > F4:
                    ll = l
                    continue
                elif Frat > F3:
                    FR = (F4 - Frat) / (F4 - F3)
                elif Frat < F2:
                    FR = (Frat - F1) / (F2 - F1)
                dp[l] = dp[l] + tonpwr_i * FR
                dpx[l] = dpx[l] + tonpwr_x * FR
                nfi = ll
                continue

        return dp, dpx

    @staticmethod
    def fan(source, theta, shield, inputs: openmdao.vectors.default_vector.DefaultVector, comp: str) -> np.ndarray:
        """
        Calculates fan noise mean-square acoustic pressure (msap) using Berton's implementation of the fan noise method.

        :param source: pyNA component computing noise sources
		:type source: Source
        :param inputs: unscaled, dimensional input variables read via inputs[key]
        :type inputs: openmdao.vectors.default_vector.DefaultVector
        :param comp: fan component (fan_inlet / fan_discharge)
        :type comp: str

        :return: msap_fan
        :rtype: np.ndarray [n_t, settings.N_f]
        """

        # Load options
        settings = source.options['settings']
        data = source.options['data']
        ac = source.options['ac']
        n_t = source.options['n_t']

        # Extract inputs
        DTt_f_star = inputs['DTt_f_star']
        mdot_f_star = inputs['mdot_f_star']
        N_f_star = inputs['N_f_star']
        A_f_star = inputs['A_f_star']
        d_f_star = inputs['d_f_star']
        M_0 = inputs['M_0']
        c_0 = inputs['c_0']
        T_0 = inputs['T_0']
        rho_0 = inputs['rho_0']

        # Initialize solution
        msap_fan = np.zeros((n_t, settings.N_f))

        # Compute fan noise
        for i in np.arange(n_t):
            ### Extract the inputs
            delta_T_fan = DTt_f_star[i] * T_0[i]  # Total temperature rise across fan [R]
            rpm = N_f_star[i] * 60 * c_0[i] / (d_f_star[i] * np.sqrt(settings.A_e))  # Shaft speed [rpm]
            M_tip_tan = (d_f_star[i] * np.sqrt(settings.A_e) / 2) * rpm * 2 * np.pi / 60 / c_0[i]  # Tangential (i.e., radius*omega) tip Mach number: ! Doesn't need correction
            mdot_fan = mdot_f_star[i] * rho_0[i] * c_0[i] * settings.A_e  # Airflow [kg/s]
            bpf = rpm * ac.B_fan / 60. / (1 - M_0[i] * np.cos(theta[i] * np.pi / 180))  # Blade passing frequency, [Hz]
            flow_M = mdot_fan / (rho_0[i] * A_f_star[i] * settings.A_e * c_0[i])  # Fan face flow Mach number (assumes ambient and fan face static densities are equal): !!!!!!!
            M_tip = (M_tip_tan ** 2 + flow_M ** 2) ** 0.5  # Relative (i.e., helical) tip Mach number: ! Doesn't need correction

            # Temperature-flow power base term:
            rho_sl = 1.22514
            c_sl = 340.29395
            if settings.fan_BB_method == 'kresja':
                tsqem = 10 * np.log10((delta_T_fan * 1.8) ** 4 * 2.20462 * mdot_fan / (1 - M_0[i] * np.cos(theta[i] * np.pi / 180)) ** 4 / settings.r_0 ** 2 / (rho_sl ** 2 * c_sl ** 4))
            else:  # All other methods:
                tsqem = 10 * np.log10((delta_T_fan * 1.8) ** 2 * 2.20462 * mdot_fan / (1 - M_0[i] * np.cos(theta[i] * np.pi / 180)) ** 4 / settings.r_0 ** 2 / (rho_sl ** 2 * c_sl ** 4))

            # Calculate individual noise components
            dcp = np.zeros(settings.N_f)
            if comp == 'fan_inlet':
                bblv_I = Fan.inlet_broadband(settings, theta[i], M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
                tonlv_I = Fan.inlet_tones(settings, theta[i], M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
                bblv_D = 0
                tonlv_X = 0
            elif comp == 'fan_discharge':
                bblv_I = 0
                tonlv_I = 0
                bblv_D = Fan.discharge_broadband(settings, theta[i], M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
                tonlv_X = Fan.discharge_tones(settings, theta[i], M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
            else:
                raise ValueError('Invalid component specified. Specify "fan_inlet" or "fan discharge".')

            # Compute combination tones
            if settings.combination_tones:
                dcp = Fan.combination_tones(settings, data.f, theta[i], M_tip, bpf, tsqem)

            # Calculate if cut-off happens (1) or not (0)
            i_cut = Fan.calculate_cutoff(M_tip_tan, ac.B_fan, ac.V_fan)

            # Assign tones_to bands
            dp, dpx = Fan.calculate_harmonics(settings, data.f, theta[i], tonlv_I, tonlv_X, i_cut, M_tip, bpf,comp)

            # Final calculations;  cycle through frequencies and assign values:
            if settings.fan_BB_method == 'allied_signal':
                # Eqn 2 or Figure 3A:
                # if data.f[j] / bpf < 2:
                # FLOGinlet,exit = -10 * np.log10(np.exp(-0.35 * (np.log(data.f[j] / bpf / 2.0) / np.log(2.2)) ** 2))
                FLOGinlet = 2.445096095 * (np.log(data.f / bpf / 2)) ** 2
                # elif data.f[j] / bpf > 2:
                # FLOGinlet,exit = -10 * np.log10(np.exp(-2.0 * (np.log(data.f[j] / bpf / 2.0) / np.log(2.2)) ** 2))
                FLOGinlet[data.f/bpf > 2] = (13.97197769 * (np.log(data.f / bpf / 2)) ** 2)[data.f/bpf > 2]
                FLOGexit = FLOGinlet
            elif settings.fan_BB_method == 'kresja':
                # Eqn 2 or Figure 3A:
                # FLOGinlet = -10 * np.log10(np.exp(-0.5 * (np.log(data.f[j] / bpf / 4) / np.log(2.2)) ** 2))
                # Which may be simplified as:
                FLOGinlet = 3.4929944 * (np.log(data.f / bpf / 4)) ** 2
                # FLOGexit = -10 * np.log10(np.exp(-0.5 * (np.log(data.f[j] / bpf / 2.5) / np.log(2.2)) ** 2))
                # Which may be simplified as:
                FLOGexit = 3.4929944 * (np.log(data.f / bpf / 2.5)) ** 2
            else:
                # For the original or the GE large fan methods:
                # Eqn 2 or Figure 3A:
                # FLOGinlet,exit = -10 * np.log10(np.exp(-0.5 * (np.log(data.f[j] / bpf / 2.5) / np.log(2.2)) ** 2))
                # Which may be simplified as:
                FLOGinlet = 3.4929944 * (np.log(data.f / bpf / 2.5)) ** 2
                FLOGexit = FLOGinlet

            # Add frequency distribution to the fan broadband noise and add the tones
            if comp == 'fan_inlet':
                msap_j = 10 ** (0.1 * (bblv_I - FLOGinlet))
                msap_j = msap_j + dp
            elif comp == 'fan_discharge':
                msap_j = 10 ** (0.1 * (bblv_D - FLOGexit))
                msap_j = msap_j + dpx
            else:
                raise ValueError('Invalid component specified.')

            # Add inlet combination tones if needed:
            if M_tip > 1 and settings.combination_tones == True:
                msap_j = msap_j + dcp

            # Multiply for number of engines
            msap_j = msap_j * ac.n_eng

            # Fan liner suppression
            if settings.fan_liner_suppression:
                if comp == 'fan_inlet':
                    supp = data.supp_fi_f(theta[i], data.f).reshape(settings.N_f,)
                elif comp == 'fan_discharge':
                    supp = data.supp_fd_f(theta[i], data.f).reshape(settings.N_f, )
                else:
                    raise ValueError('Invalid fan component.')
                msap_j = supp * msap_j

            # Fan inlet shielding
            if settings.shielding and comp == 'fan_inlet':
                msap_j = msap_j / (10 ** (shield[i, :] / 10.))
            msap_fan[i, :] = msap_j

        return msap_fan

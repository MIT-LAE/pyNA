import numpy as np
import pandas as pd
from scipy import interpolate
import pdb


class Tables:
    
    def __init__(self, settings) -> None:
        """
        Initialize Data class.

        :return: None
        """

        # Load data tables
        Tables.load_atmospheric_absorption_tables(self, settings=settings)
        Tables.load_faddeeva_tables(self, settings=settings)
        Tables.load_fan_suppression_tables(self, settings=settings)
        Tables.load_airframe_hsr_suppression_tables(self, settings=settings)
        Tables.load_sae_arp876_tables(self, settings=settings)
        Tables.load_noy_tables(self, settings=settings)
        Tables.load_a_weighting_tables(self)

        # Initialize verification data
        self.verification_source = dict()
        self.verification_source_supp = dict()
        self.verification_trajectory = dict()

    def load_atmospheric_absorption_tables(self, settings) -> None:
        """
        Load atmospheric absorption coefficient table.

        :return: None
        """

        # Load atmospheric absorption coefficient table
        # Source: verification noise assessment data set of NASA STCA (Berton et al., 2019)
        self.abs = pd.read_csv(settings['pyna_directory']+'/data/isa/atmospheric_absorption.csv', skiprows=1).values[:, 1:]
        self.abs_freq = np.array(pd.read_csv(settings['pyna_directory']+'/data/isa/atmospheric_absorption.csv', nrows=1).values[0][1:], dtype=float)
        self.abs_alt = pd.read_csv(settings['pyna_directory']+'/data/isa/atmospheric_absorption.csv', skiprows=1).values[:, 0]
        self.abs_f = interpolate.interp2d(self.abs_freq, self.abs_alt, self.abs, kind='linear')

        return None

    def load_faddeeva_tables(self, settings) -> None:
        """
        
        """

        # Load Faddeeva tables
        faddeeva_data_real = pd.read_csv(settings['pyna_directory']+'/data/propagation/Faddeeva_real_small.csv').values
        faddeeva_data_imag = pd.read_csv(settings['pyna_directory']+'/data/propagation/Faddeeva_imag_small.csv').values
        self.faddeeva_itau_re = faddeeva_data_real[0, 1:]
        self.faddeeva_itau_im = faddeeva_data_real[1:, 0]
        self.faddeeva_real = faddeeva_data_real[1:,1:]
        self.faddeeva_imag = faddeeva_data_imag[1:,1:]

        return None

    def load_fan_suppression_tables(self, settings) -> None:
        """
        Load the noise suppression tables for fan inlet and discharge source noise suppression

        :return: None
        """

        # Load noise source suppression data
        # Source: verification noise assessment data set of NASA STCA (Berton et al., 2019)
        self.supp_fi = pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_inlet_suppression.csv', skiprows=1).values[:, 1:]
        self.supp_fi_angles = np.array(pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_inlet_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_fi_freq = pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_inlet_suppression.csv',skiprows=1).values[:, 0]
        self.supp_fi_f = interpolate.interp2d(self.supp_fi_angles, self.supp_fi_freq, self.supp_fi, kind='linear')

        self.supp_fd = pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_discharge_suppression.csv', skiprows=1).values[:, 1:]
        self.supp_fd_angles = np.array(pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_discharge_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_fd_freq = pd.read_csv(settings['pyna_directory']+'/data/sources/fan/liner_discharge_suppression.csv',skiprows=1).values[:, 0]
        self.supp_fd_f = interpolate.interp2d(self.supp_fd_angles, self.supp_fd_freq, self.supp_fd, kind='linear')

        return None

    def load_airframe_hsr_suppression_tables(self, settings) -> None:
        """
        
        """

        self.supp_af = pd.read_csv(settings['pyna_directory']+'/data/sources/airframe/hsr_suppression.csv', skiprows=1).values[:,1:]
        self.supp_af_angles = np.array(pd.read_csv(settings['pyna_directory']+'/data/sources/airframe/hsr_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_af_freq = pd.read_csv(settings['pyna_directory']+'/data/sources/airframe/hsr_suppression.csv', skiprows=1).values[:, 0]
        self.supp_af_f = interpolate.interp2d(self.supp_af_angles, self.supp_af_freq, self.supp_af, kind='linear')

        return None

    def load_sae_arp876_tables(self, settings) -> None:
        """
        Load the jet source noise model data.

        :return: None
        """

        # Polar directivity level D
        # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
        self.jet_D = pd.read_csv(settings['pyna_directory'] + '/data/sources/jet/directivity_function.csv').values
        self.jet_D_angles = self.jet_D[0, 1:]
        self.jet_D_velocity = self.jet_D[1:, 0]
        self.jet_D = self.jet_D[1:, 1:]
        self.jet_D_f = interpolate.interp2d(self.jet_D_angles,
                                            self.jet_D_velocity,
                                            self.jet_D, kind='linear')

        # Strouhal number correction factor xi
        # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
        self.jet_xi = pd.read_csv(settings['pyna_directory'] + '/data/sources/jet/strouhal_correction.csv').values
        self.jet_xi_angles = self.jet_xi[0, 1:]
        self.jet_xi_velocity = self.jet_xi[1:, 0]
        self.jet_xi = self.jet_xi[1:,1:]
        self.jet_xi_f = interpolate.interp2d(self.jet_xi_angles,
                                             self.jet_xi_velocity,
                                             self.jet_xi, kind='linear')

        # Spectral level F
        # Source: Zorumski report 1982 part 1. Chapter 8.4 Table VI
        self.jet_F = np.load(settings['pyna_directory'] + '/data/sources/jet/spectral_function_extended_T.npy')
        self.jet_F_angles = np.array([0., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
        self.jet_F_temperature = np.array([0., 1., 2., 2.5, 3., 3.5, 4., 5., 6., 7.])
        self.jet_F_velocity = np.array([-0.4, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.4])
        self.jet_F_strouhal = np.array([-2., -1.6, -1.3, -1.15, -1., -0.824, -0.699, -0.602, -0.5, -0.398, -0.301, -0.222, 0., 0.477, 1., 1.6, 1.7, 2.5])
        self.jet_F_f = interpolate.RegularGridInterpolator((self.jet_F_angles,
                                                            self.jet_F_temperature,
                                                            self.jet_F_velocity,
                                                            self.jet_F_strouhal),
                                                            self.jet_F)

        return None

    def load_noy_tables(self, settings) -> None:
        """
        Load noy tables for tone-corrected perceived noise level (pnlt) computation.

        :return: None
        """

        # Load noise level computation tabular data
        # Source: ICAO Annex 16 Volume 1 (Edition 8) Table A2-3.
        # self.Noy = pd.read_csv(settings['pyna_directory']+'/data/levels/pnlt_noy_weighting.csv')

        noy = pd.read_csv(settings['pyna_directory']+'/data/levels/spl_noy.csv').values
        self.noy_spl = noy[1:, 0]
        self.noy_freq = noy[0, 1:]
        self.noy = noy[1:,1:]

        self.noy_f = interpolate.interp2d(self.noy_freq,
                                          self.noy_spl,
                                          self.noy, kind='linear')

        return None

    def load_a_weighting_tables(self) -> None:
        """
        Load A-weighting coefficients for A-SPL computation.

        :return: None
        """

        self.aw_freq = np.array([10, 20, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
        self.aw_db = np.array([-70.4, -50.4, -34.5, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.7, -9.3])

        return None

 
        """
        Load verification data for noise source spectral and directional distributions.

        :param components: list of components to run
        :type components: list

        :return: (data_val, data_val_s)
        :rtype: (dict, dict)
        """

        for comp in components:

            if comp == 'core':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Core Module Source.xlsx', sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Core Module Source.xlsx', sheet_name='Suppressed').values

            elif comp == 'jet_mixing':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Jet Module Source.xlsx', sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Jet Module Source.xlsx', sheet_name='Suppressed').values

            elif comp == 'inlet_BB':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Full Inlet BB').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Suppressed Inlet BB').values

            elif comp == 'discharge_BB':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Full Discharge BB').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Suppressed Discharge BB').values

            elif comp == 'inlet_RS':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Full Inlet RS').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Suppressed Inlet RS').values

            elif comp == 'discharge_RS':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Full Discharge RS').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Suppressed Discharge RS').values

            elif comp == 'fan_inlet':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Inlet Full').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Inlet Suppressed').values

            elif comp == 'fan_discharge':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Discharge Full').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Discharge Suppressed').values

            elif comp == 'airframe':
                self.verification_source[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Airframe Module Source.xlsx', sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Airframe Module Source.xlsx', sheet_name='Suppressed').values

        return None
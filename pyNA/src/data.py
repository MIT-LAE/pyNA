import numpy as np
import pandas as pd
from scipy import interpolate
from pyNA.src.settings import Settings
from dataclasses import dataclass
import pdb


@dataclass()
class Data:

    """
    Data class containing the following parameters:

    * ``f``:                  1/3-rd octave frequency bands [Hz]
    * ``f_sb``:               sub-bands of the 1/3rd octave frequency bands [Hz]

    * ``abs``:                table with atmospheric absorption coefficients
    * ``abs_freq``:           frequencies of atmospheric absorption coefficient table
    * ``abs_alt``:            altitudes of atmospheric absorption coefficient table
    * ``abs_f``:              interpolation function for atmospheric absorption coefficient

    * ``jet_D``:              table with jet mixing noise directivity dependency
    * ``jet_D_angles``:       table with jet mixing noise directivity dependency: directivity angles
    * ``jet_D_velocity``:     table with jet mixing noise directivity dependency: jet velocity ratio
    * ``jet_D_f``:            interpolation function for jet mixing noise directivity dependency
    * ``jet_xi``:             table with jet mixing noise Strouhal number correction factor
    * ``jet_xi_angles``:      table with jet mixing noise Strouhal number correction factor: directivity angles
    * ``jet_xi_velocity``:    table with jet mixing noise Strouhal number correction factor: jet velocity ratio
    * ``jet_xi_f``:           interpolation function for jet mixing noise Strouhal number correction factor
    * ``jet_F``:              table with jet mixing noise spectral dependency
    * ``jet_F_angles``:       table with jet mixing noise spectral dependency: directivity angles
    * ``jet_F_temperature``:  table with jet mixing noise spectral dependency: jet temperature ratio
    * ``jet_F_velocity``:     table with jet mixing noise spectral dependency: jet velocity ratio
    * ``jet_F_strouhal``:     table with jet mixing noise spectral dependency: Strouhal number
    * ``jet_F_f``:            interpolation function for jet mixing noise spectral dependency

    * ``Faddeeva_itau_re``:   table with real part of the input to the Faddeeva function (for julia ground reflections)
    * ``Faddeeva_itau_im``:   table with imaginary part of the input to the Faddeeva function (for julia ground reflections)
    * ``Faddeeva_real``:      table with real part of the Faddeeva function (for julia ground reflections)
    * ``Faddeeva_imag``:      table with imaginary part of the Faddeeva function (for julia ground reflections)

    * ``noy_spl``:            sound pressure levels of noy table
    * ``noy_freq``:           1/3-rd octave frequency bands of noy table
    * ``noy``:                noy table
    * ``noy_f``:              inteprolation function for the noy table

    * ``supp_fi``:            table with suppression factors for fan inlet noise
    * ``supp_fi_angles``:     directivity angles of suppression factors for fan inlet noise table
    * ``supp_fi_freq``:       frequencies of suppression factors for fan inlet noise table
    * ``supp_fi_f``:          interpolation function for fan inlet noise suppression coefficient
    * ``supp_fd``:            table with suppression factors for fan discharge noise
    * ``supp_fd_angles``:     directivity angles of suppression factors for fan discharge noise table
    * ``supp_fd_freq``:       frequencies of suppression factors for fan discharge noise table
    * ``supp_fd_f``:          interpolation function for fan discharge noise suppression coefficient
    * ``supp_af``:            table with HSR suppression factors for airframe noise
    * ``supp_af_angles``:     directivity angles of suppression factors for airframe noise table
    * ``supp_af_freq``:       frequencies of suppression factors for airframe noise table
    * ``supp_af_f``:          interpolation function for HSR airframe noise suppression coefficient

    * ``shield_l``:           lateral microphone airframe shielding delta dB [dB]
    * ``shield_f``:           flyover microphone airframe shielding delta dB [dB]
    * ``shield_a``:           approach microphone airframe shielding delta dB [dB]

    """

    # Frequency spectrum
    f : np.ndarray
    f_sb : np.ndarray

    abs : np.ndarray
    abs_freq : np.ndarray
    abs_alt : np.ndarray
    abs_f : interpolate.interpolate.interp2d

    jet_D: np.ndarray
    jet_D_angles: np.ndarray
    jet_D_velocity: np.ndarray
    jet_D_f: interpolate.interpolate.interp2d
    jet_xi: np.ndarray
    jet_xi_angles: np.ndarray
    jet_xi_velocity: np.ndarray
    jet_xi_f: interpolate.interpolate.interp2d
    jet_F: np.ndarray
    jet_F_angles: np.ndarray
    jet_F_temperature: np.ndarray
    jet_F_velocity: np.ndarray
    jet_F_strouhal: np.ndarray
    jet_F_f: interpolate.interpolate.RegularGridInterpolator

    Faddeeva_itau_re : np.ndarray
    Faddeeva_itau_im : np.ndarray
    Faddeeva_real : np.ndarray
    Faddeeva_imag : np.ndarray

    noy_spl : np.ndarray
    noy_freq : np.ndarray
    noy : np.ndarray
    noy_f : interpolate.interpolate.interp2d

    supp_fi : np.ndarray
    supp_fi_angles : np.ndarray
    supp_fi_freq : np.ndarray
    supp_fi_f: interpolate.interpolate.interp2d
    supp_fd : np.ndarray
    supp_fd_angles : np.ndarray
    supp_fd_freq : np.ndarray
    supp_fd_f: interpolate.interpolate.interp2d
    supp_af : np.ndarray
    supp_af_angles : np.ndarray
    supp_af_freq : np.ndarray
    supp_af_f: interpolate.interpolate.interp2d

    shield_l : np.ndarray
    shield_f : np.ndarray
    shield_a : np.ndarray

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Data class.

        :param settings: pyna settings
        :type settings: Settings

        :return: None
        """

        # Load data tables
        Data.compute_frequency_bands(self, settings=settings)
        Data.load_propagation_tables(self, settings=settings)
        Data.load_suppression_tables(self, settings=settings)
        Data.load_jet_data(self, settings=settings)
        Data.load_noy_data(self, settings=settings)
        Data.load_a_weighting_data(self)
        Data.load_shielding_time_series(self, settings=settings)

        # Initialize verification data
        self.verification_source = dict()
        self.verification_source_supp = dict()
        self.verification_trajectory = dict()

    def compute_frequency_bands(self, settings: Settings) -> None:
        """
        Compute the 1/3rd order frequency bands and with sub-bands.
            * f:    1/3rd order frequency bands
            * f_sb: frequency sub-bands

        :param settings: pyna settings.
        :type settings: Settings

        :return: None
        """

        # Load 1/3rd order frequency bands
        # Generate 1/3rd octave frequency bands [Hz]
        l_i = 16  # Starting no. of the frequency band [-]
        self.f = 10 ** (0.1 * np.linspace(1+l_i, 40, settings.N_f))

        # Calculate subband frequencies [Hz]
        # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 6-7
        # Source: Berton 2021 Simultaneous use of Ground Reflection and Lateral Attenuation Noise Models Appendix A Eq. 1
        self.f_sb = np.zeros(settings.N_b * settings.N_f)
        m = (settings.N_b - 1) / 2.
        w = 2. ** (1 / (3. * settings.N_b))
        for k in np.arange(settings.N_f):
            for h in np.arange(settings.N_b):
                self.f_sb[k * settings.N_b + h] = w ** (h - m) * self.f[k]

        return None

    def load_propagation_tables(self, settings: Settings) -> None:
        """
        Load atmospheric absorption coefficient table.

        :param settings: pyna settings.
        :type settings: Settings

        :return: None
        """

        # Load atmospheric absorption coefficient table
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.abs = pd.read_csv(settings.pyNA_directory+'/data/isa/atmospheric_absorption.csv', skiprows=1).values[:, 1:]
        self.abs_freq = np.array(pd.read_csv(settings.pyNA_directory+'/data/isa/atmospheric_absorption.csv', nrows=1).values[0][1:], dtype=float)
        self.abs_alt = pd.read_csv(settings.pyNA_directory+'/data/isa/atmospheric_absorption.csv', skiprows=1).values[:, 0]
        self.abs_f = interpolate.interp2d(self.abs_freq, self.abs_alt, self.abs, kind='linear')

        # Load Faddeeva tables
        faddeeva_data_real = pd.read_csv(settings.pyNA_directory+'/data/propagation/Faddeeva_real_small.csv').values
        faddeeva_data_imag = pd.read_csv(settings.pyNA_directory+'/data/propagation/Faddeeva_imag_small.csv').values
        self.Faddeeva_itau_re = faddeeva_data_real[0, 1:]
        self.Faddeeva_itau_im = faddeeva_data_real[1:, 0]
        self.Faddeeva_real = faddeeva_data_real[1:,1:]
        self.Faddeeva_imag = faddeeva_data_imag[1:,1:]

        return None

    def load_suppression_tables(self, settings: Settings) -> None:
        """
        Load the noise suppression tables for:
            * fan inlet source noise suppression
            * fan discharge source noise suppression
            * airframe noise suppression (high-speed research program)

        :param settings: pyna settings.
        :type settings: Settings

        :return: None
        """

        # Load noise source suppression data
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        self.supp_fi = pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_inlet_suppression.csv', skiprows=1).values[:, 1:]
        self.supp_fi_angles = np.array(pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_inlet_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_fi_freq = pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_inlet_suppression.csv',skiprows=1).values[:, 0]
        self.supp_fi_f = interpolate.interp2d(self.supp_fi_angles, self.supp_fi_freq, self.supp_fi, kind='linear')

        self.supp_fd = pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_discharge_suppression.csv', skiprows=1).values[:, 1:]
        self.supp_fd_angles = np.array(pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_discharge_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_fd_freq = pd.read_csv(settings.pyNA_directory+'/data/sources/fan/liner_discharge_suppression.csv',skiprows=1).values[:, 0]
        self.supp_fd_f = interpolate.interp2d(self.supp_fd_angles, self.supp_fd_freq, self.supp_fd, kind='linear')

        self.supp_af = pd.read_csv(settings.pyNA_directory+'/data/sources/airframe/hsr_suppression.csv', skiprows=1).values[:,1:]
        self.supp_af_angles = np.array(pd.read_csv(settings.pyNA_directory+'/data/sources/airframe/hsr_suppression.csv', skiprows=0).values[0,1:], dtype=float)
        self.supp_af_freq = pd.read_csv(settings.pyNA_directory+'/data/sources/airframe/hsr_suppression.csv', skiprows=1).values[:, 0]
        self.supp_af_f = interpolate.interp2d(self.supp_af_angles, self.supp_af_freq, self.supp_af, kind='linear')

        return None

    def load_jet_data(self, settings: Settings) -> None:
        """
        Load the jet source noise model data.

        :param settings: pyna settings
        :type settings: Settings

        :return: None
        """

        # Polar directivity level D
        # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
        self.jet_D = pd.read_csv(settings.pyNA_directory + '/data/sources/jet/directivity_function.csv').values
        self.jet_D_angles = self.jet_D[0, 1:]
        self.jet_D_velocity = self.jet_D[1:, 0]
        self.jet_D = self.jet_D[1:, 1:]
        self.jet_D_f = interpolate.interp2d(self.jet_D_angles,
                                            self.jet_D_velocity,
                                            self.jet_D, kind='linear')

        # Strouhal number correction factor xi
        # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
        self.jet_xi = pd.read_csv(settings.pyNA_directory + '/data/sources/jet/strouhal_correction.csv').values
        self.jet_xi_angles = self.jet_xi[0, 1:]
        self.jet_xi_velocity = self.jet_xi[1:, 0]
        self.jet_xi = self.jet_xi[1:,1:]
        self.jet_xi_f = interpolate.interp2d(self.jet_xi_angles,
                                             self.jet_xi_velocity,
                                             self.jet_xi, kind='linear')

        # Spectral level F
        # Source: Zorumski report 1982 part 1. Chapter 8.4 Table VI
        self.jet_F = np.load(settings.pyNA_directory + '/data/sources/jet/spectral_function_extended_T.npy')
        self.jet_F_angles = np.array([0, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
        self.jet_F_temperature = np.array([0, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 7])
        self.jet_F_velocity = np.array([-0.4, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.4])
        self.jet_F_strouhal = np.array([-2, -1.6, -1.3, -1.15, -1, -0.824, -0.699, -0.602, -0.5, -0.398, -0.301, -0.222, 0, 0.477, 1, 1.6, 1.7, 2.5])
        self.jet_F_f = interpolate.RegularGridInterpolator((self.jet_F_angles,
                                                            self.jet_F_temperature,
                                                            self.jet_F_velocity,
                                                            self.jet_F_strouhal),
                                                            self.jet_F)

        return None

    def load_noy_data(self, settings: Settings) -> None:
        """
        Load noy tables for tone-corrected perceived noise level (pnlt) computation.

        :param settings: pyna settings.
        :type settings: Settings

        :return: None
        """

        # Load noise level computation tabular data
        # Source: ICAO Annex 16 Volume 1 (Edition 8) Table A2-3.
        # self.Noy = pd.read_csv(settings.pyNA_directory+'/data/levels/pnlt_noy_weighting.csv')

        noy = pd.read_csv(settings.pyNA_directory+'/data/levels/spl_noy.csv').values
        self.noy_spl = noy[1:, 0]
        self.noy_freq = noy[0, 1:]
        self.noy = noy[1:,1:]

        self.noy_f = interpolate.interp2d(self.noy_freq,
                                          self.noy_spl,
                                          self.noy, kind='linear')

        return None


    def load_a_weighting_data(self) -> None:
        """
        Load A-weighting coefficients for A-SPL computation.

        :param settings: pyna settings.
        :type settings: Settings

        :return: None
        """

        self.aw_freq = np.array([10, 20, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
        self.aw_db = np.array([-70.4, -50.4, -34.5, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.7, -9.3])

        return None

    def load_trajectory_verification_data(self, settings: Settings) -> None:
        """
        Loads the verification data of the NASA STCA noise assessment (Berton et al., 2019).

        :param settings: pyNA settings
        :type settings: Settings

        :return: data_val
        :rtype: pd.DataFrame

        """

        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        if settings.case_name == 'nasa_stca_standard':
            # Check if all sources
            if settings.all_sources:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Total.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='total', engine="openpyxl")

            # Check if individual components
            elif settings.jet_mixing:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Jet CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='jet', engine="openpyxl")

            elif settings.core:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Core CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='core', engine="openpyxl")

            elif settings.airframe:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Airframe CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='airframe', engine="openpyxl")

            elif settings.fan_inlet:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Fan Inlet CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='fan inlet', engine="openpyxl")

            elif settings.fan_discharge:
                for i, observer in enumerate(settings.observer_lst):
                    if observer in ['lateral', 'flyover']:
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/55t-Depart-Standard-Fan Discharge CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.verification_trajectory[observer] = pd.read_excel(settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Approach Levels.xlsx', sheet_name='fan discharge', engine="openpyxl")
            else:
                raise ValueError('Invalid noise component for trajectory validation. Specify core/jet mixing/fan inlet/fan discharge/airframe')
        else:
            raise ValueError('No validation data available for this case name. Specify nasa_stca_standard.')

        return None

    def load_source_verification_data(self, settings: Settings, components: list):
        """
        Load verification data for noise source spectral and directional distributions.

        :param settings: pyNA settings
        :type settings: Settings
        :param components: list of components to run
        :type components: list

        :return: (data_val, data_val_s)
        :rtype: (dict, dict)
        """

        for comp in components:
            if comp == 'core':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Core Module Source.xlsx',
                    sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Core Module Source.xlsx',
                    sheet_name='Suppressed').values

            elif comp == 'jet_mixing':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Jet Module Source.xlsx',
                    sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Jet Module Source.xlsx',
                    sheet_name='Suppressed').values

            elif comp == 'inlet_BB':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Full Inlet BB').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Suppressed Inlet BB').values

            elif comp == 'discharge_BB':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Full Discharge BB').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Suppressed Discharge BB').values

            elif comp == 'inlet_RS':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Full Inlet RS').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Suppressed Inlet RS').values

            elif comp == 'discharge_RS':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Full Discharge RS').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Suppressed Discharge RS').values

            elif comp == 'fan_inlet':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Fan Inlet Full').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Fan Inlet Suppressed').values

            elif comp == 'fan_discharge':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Fan Discharge Full').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Fan Module Source.xlsx',
                    sheet_name='Fan Discharge Suppressed').values

            elif comp == 'airframe':
                self.verification_source[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Airframe Module Source.xlsx',
                    sheet_name='Full').values
                self.verification_source_supp[comp] = pd.read_excel(
                    settings.pyNA_directory + '/cases/' + settings.case_name + '/validation/Airframe Module Source.xlsx',
                    sheet_name='Suppressed').values

        return None

    def load_shielding_time_series(self, settings: Settings):

        self.shield_l = []
        self.shield_f = []
        self.shield_a = []

        if settings.case_name == 'nasa_stca_standard':

            for i, observer in enumerate(settings.observer_lst):

                if observer == 'lateral':
                    self.shield_l = pd.read_csv(settings.pyNA_directory+'/cases/'+settings.case_name + '/shielding/shielding_l.csv').values[:,1:]

                elif observer == 'flyover':
                    self.shield_f = pd.read_csv(settings.pyNA_directory+'/cases/'+settings.case_name + '/shielding/shielding_f.csv').values[:,1:]

                elif observer == 'approach':
                    self.shield_a = pd.read_csv(settings.pyNA_directory+'/cases/'+settings.case_name + '/shielding/shielding_a.csv').values[:,1:]


import jax.numpy as jnp
import pandas as pd
from scipy import interpolate
import pyNA
import pdb


class Tables:
    
    def __init__(self, settings) -> None:
        """
        Initialize Data class.

        :return: None
        """

        self.source = Tables.Source(settings=settings)
        self.propagation = Tables.Propagation()
        self.levels = Tables.Levels()

    class Source:
        
        def __init__(self, settings) -> None:
            
            self.fan = Tables.Source.Fan(settings=settings)
            self.core = Tables.Source.Core()
            self.jet = Tables.Source.Jet()
            self.airframe = Tables.Source.Airframe()

        class Fan:
            def __init__(self, settings) -> None:
                
                Tables.Source.Fan.get_fan_heidman_tables(self, settings=settings)
                Tables.Source.Fan.get_fan_heidman_liner_suppression_tables(self)

            def get_fan_heidman_tables(self, settings):

                # Inlet broadband
                if settings['fan_BB_method'] == 'kresja':
                    self.f3ib_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190.])
                    self.f3ib_data =  jnp.array([-0.5, -1., -1.25, -1.41, -1.4, -2.2, -4.5, -8.5, -13., -18.5, -24., -30., -36., -42., -48., -54., -60., -66., -73., -66.])
                else:
                    self.f3ib_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 180., 250.])
                    self.f3ib_data = jnp.array([-2., -1., 0., 0., 0., -2., -4.5, -7.5, -11., -15., -19.5, -25., -63.5, -25.])

                # Discharge broadband
                if settings['fan_BB_method'] == 'allied_signal':
                    self.f3db_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190.])
                    self.f3db_data = jnp.array([0., -29.5, -26., -22.5, -19., -15.5, -12., -8.5, -5., -3.5, -2.5, -2., -1.3, 0., -3., -7., -11., -15., -20., -15.])
                elif settings['fan_BB_method'] == 'kresja':
                    self.f3db_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190.])
                    self.f3db_data = jnp.array([-30., -25., -20.8, -19.5, -18.4, -16.7, -14.5, -12., -9.6, -6.9, -4.5, -1.8, -0.3, 0.5, 0.7, -1.9,-4.5, -9., -15., -9.])
                else:
                    self.f3db_theta = jnp.array([0., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190.])
                    self.f3db_data = jnp.array([-41.6, -15.8, -11.5, -8., -5., -2.7, -1.2, -0.3, 0., -2., -6., -10., -15., -20., -15.])
                    
                # Inlet tones
                if settings['fan_RS_method'] == 'allied_signal':
                    self.f3ti_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                    self.f3ti_data = jnp.array([-3., -1.5, -1.5, -1.5, -1.5, -2., -3., -4., -6., -9., -12.5, -16., -19.5, -23., -26.5, -30., -33.5, -37., -40.5])
                elif settings['fan_RS_method'] == 'kresja':
                    self.f3ti_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                    self.f3ti_data = jnp.array([-3., -1.5, 0., 0., 0., -1.2, -3.5, -6.8, -10.5, -15.5, -19., -25., -32., -40., -49., -59., -70., -80., -90.])
                else:
                    self.f3ti_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 180., 260.])
                    self.f3ti_data = jnp.array([-3., -1.5, 0., 0., 0., -1.2, -3.5, -6.8, -10.5, -14.5, -19., -55., -19.])

                # Discharge tones
                if settings['fan_RS_method'] == 'allied_signal':
                    self.f3td_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                    self.f3td_data = jnp.array([-34., -30., -26., -22., -18., -14., -10.5, -6.5, -4., -1., 0., 0., 0., 0., -1., -3.5, -7., -11., -16.])
                elif settings['fan_RS_method'] == 'kresja':
                    self.f3td_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                    self.f3td_data = jnp.array([-50., -41., -33., -26., -20.6, -17.9, -14.7, -11.2, -9.3, -7.1, -4.7, -2., 0., 0.8, 1., -1.6, -4.2, -9., -15.])
                elif settings['fan_RS_method'] in ['original', 'geae']:
                    self.f3td_theta = jnp.array([0., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190])
                    self.f3td_data = jnp.array([-39., -15., -11., -8., -5., -3., -1., 0., 0., -2., -5.5, -9., -13., -18., -13.])

                # Combination tones
                if settings['fan_RS_method'] == 'original':
                    self.f2ct_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 180., 270.])
                    self.f2ct_data = jnp.array([-9.5, -8.5, -7., -5., -2., 0., 0., -3.5, -7.5, -9., -13.5, -9.])
                elif settings['fan_RS_method'] == 'allied_signal':
                    self.f2ct_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 270.])
                    self.f2ct_data = jnp.array([-5.5, -4.5, -3., -1.5, 0., 0., 0., 0., -2.5, -5., -6., -6.9, -7.9, -8.8, -9.8, -10.7, -11.7, -12.6, -13.6, -6.])
                elif settings['fan_RS_method'] == 'geae':
                    self.f2ct_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 180., 270.])
                    self.f2ct_data = jnp.array([-9.5, -8.5, -7., -5., -2., 0., 0., -3.5, -7.5, -9., -13.5, -9.])
                elif settings['fan_RS_method'] == 'kresja':
                    self.f2ct_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190.])
                    self.f2ct_data = jnp.array([-28., -23., -18., -13., -8., -3., 0., -1.3, -2.6, -3.9, -5.2, -6.5, -7.9, -9.4, -11., -12.7, -14.5, -16.4, -18.4])

                # Flight clean-up effects
                self.fe_theta = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                self.fe_takeoff_1_data  = jnp.array([4.8, 4.8, 5.5, 5.5, 5.3, 5.3, 5.1, 4.4, 3.9, 2.6, 2.3, 1.8, 2.1, 1.7, 1.7, 2.6, 3.5, 3.5, 3.5])
                self.fe_takeoff_2_data  = jnp.array([5.8, 5.8, 3.8, 5.3, 6.4, 3.5, 3.0, 2.1, 2.1, 1.1, 1.4, 0.9, 0.7, 0.7, 0.4, 0.6, 0.8, 0.8, 0.8])
                self.fe_approach_1_data = jnp.array([5.6, 5.6, 5.8, 4.7, 4.6, 4.9, 5.1, 2.9, 3.2, 1.6, 1.6, 1.8, 2.1, 2.4, 2.2, 2.0, 2.8, 2.8, 2.8])
                self.fe_approach_2_data = jnp.array([5.4, 5.4, 4.3, 3.4, 4.1, 2.0, 2.9, 1.6, 1.3, 1.5, 1.1, 1.4, 1.5, 1.0, 1.8, 1.6, 1.6, 1.6, 1.6])

                return 

            def get_fan_heidman_liner_suppression_tables(self) -> None:
                """
                Load the noise suppression tables for fan inlet and discharge source noise suppression

                :return: None
                """

                # Load noise source suppression data
                # Source: verification noise assessment data set of NASA STCA (Berton et al., 2019)
                data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/fan/liner_inlet_suppression.csv')
                self.supp_fi_theta = data.values[0,1:]
                self.supp_fi_f = data.values[1:,0]
                self.supp_fi_data = data.values[1:, 1:] 

                self.supp_fi_interp = interpolate.interp2d(self.supp_fi_theta, self.supp_fi_f, self.supp_fi_data, kind='linear')

                data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/fan/liner_discharge_suppression.csv')
                self.supp_fd_theta = data.values[0, 1:]
                self.supp_fd_f = data.values[1:, 0]
                self.supp_fd_data = data.values[1:,1:]
                
                self.supp_fd_interp = interpolate.interp2d(self.supp_fd_theta, self.supp_fd_f, self.supp_fd_data, kind='linear')

                return None

        class Core:
            def __init__(self) -> None:

                Tables.Source.Core.get_core_emmerling_tables(self)

            def get_core_emmerling_tables(self) -> None:
                
                self.D_theta = jnp.linspace(0, 180, 19)
                self.D_data = jnp.array([-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,-0.19,-0.51, -0.8, -0.9])
                
                self.S_log10ffp = jnp.linspace(-1.1, 1.6, 28)
                self.S_data = jnp.array([-3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97, -1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2])
                
                return None

        class Jet:
            def __init__(self) -> None:
                
                Tables.Source.Jet.get_jet_mixing_sae876_tables(self)
                Tables.Source.Jet.get_jet_shock_sae876_tables(self)

            def get_jet_mixing_sae876_tables(self) -> None:
                """
                Load the jet source noise model data.

                :return: None
                """

                # Density exponent (omega)
                # Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
                self.omega_log10Vjc0 = jnp.array([-0.45, -0.4, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 2.0])
                self.omega_data = jnp.array([-1.00, -0.9, -0.76, -0.58, -0.41, -0.22,  0.00,  0.22,  0.50, 0.77, 1.07, 1.39, 1.74, 1.95, 2.00, 2.0])

                # Power deviation factor (P)
                # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
                self.P_log10Vjc0 = jnp.array([-0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.1, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
                self.P_data = jnp.array([-0.13, -0.13, -0.13, -0.13, -0.13, -0.12, -0.1, -0.05, 0, 0.10, 0.21, 0.32, 0.41, 0.43, 0.41, 0.31, 0.14])

                # Polar directivity level (D)
                # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
                data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/jet/directivity_function.csv').values
                self.D_theta = data[0, 1:]
                self.D_log10Vjc0 = data[1:, 0]
                self.D_data = data[1:, 1:]

                self.D_interp = interpolate.interp2d(self.D_theta, self.D_log10Vjc0, self.D_data, kind='linear')

                # Strouhal number correction factor (xi)
                # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
                data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/jet/strouhal_correction.csv').values
                self.xi_theta = data[0, 1:]
                self.xi_Vjc0 = data[1:, 0]
                self.xi_data = data[1:,1:]

                self.xi_interp = interpolate.interp2d(self.xi_theta, self.xi_Vjc0, self.xi_data, kind='linear')

                # Spectral level F
                # Source: Zorumski report 1982 part 1. Chapter 8.4 Table VI
                self.F_theta = jnp.array([0., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.])
                self.F_TjT0 = jnp.array([0., 1., 2., 2.5, 3., 3.5, 4., 5., 6., 7.])
                self.F_log10Vjc0 = jnp.array([-0.4, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.4])
                self.F_strouhal = jnp.array([-2., -1.6, -1.3, -1.15, -1., -0.824, -0.699, -0.602, -0.5, -0.398, -0.301, -0.222, 0., 0.477, 1., 1.6, 1.7, 2.5])
                self.F_data = jnp.load(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/jet/spectral_function_extended_T.npy')

                self.F_interp = interpolate.RegularGridInterpolator((self.F_theta, self.F_TjT0, self.F_log10Vjc0, self.F_strouhal), self.F_data)

                # Calculate forward velocity index (m_theta)
                # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
                self.m_theta_theta = jnp.linspace(0., 180., 19)
                self.m_theta_data = jnp.array([3., 1.65, 1.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.4, 1, 1.9, 3, 4.7, 7, 8.5, 8.5, 8.5, 8.5, 8.5])

                return None

            def get_jet_shock_sae876_tables(self) -> None:

                """
                """

                self.C_log10sigma = jnp.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
                self.C_data = jnp.array([0.703, 0.703, 0.71, 0.714, 0.719, 0.724, 0.729, 0.735, 0.74, 0.74, 0.74, 0.735, 0.714,0.681, 0.635, 0.579, 0.52, 0.46, 0.4, 0.345, 0.29, 0.235, 0.195, 0.15, 0.1, 0.06, 0.03, 0.015])

                self.H_log10sigma = jnp.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
                self.H_data = jnp.array([-5.73, -5.35, -4.97, -4.59, -4.21, -3.83, -3.45, -3.07, -2.69, -2.31, -1.94, -1.59, -1.33,-1.1, -0.94, -0.88, -0.91, -0.99, -1.09, -1.17, -1.3, -1.42, -1.55, -1.67, -1.81, -1.92, -2.06, -2.18, -2.3, -2.42, -2.54, -2.66, -2.78, -2.9])

                return None

        class Airframe:
            def __init__(self) -> None:
                
                Tables.Source.Airframe.get_airframe_hsr_suppression_tables(self)

            def get_airframe_hsr_suppression_tables(self) -> None:
                
                """  
                """

                data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/sources/airframe/hsr_suppression.csv')
                self.hsr_supp_theta = data.values[0,1:]
                self.hsr_supp_f = data.values[1:,0]
                self.hsr_supp_data = data.values[1:,1:]
                
                self.supp_af_interp = interpolate.interp2d(self.hsr_supp_theta, self.hsr_supp_f, self.hsr_supp_data, kind='linear')

                return None


    class Propagation:

        def __init__(self) -> None:

            Tables.Propagation.get_atmospheric_absorption_tables(self)
            Tables.Propagation.get_faddeeva_tables(self)
        
        def get_atmospheric_absorption_tables(self) -> None:
            
            """
            Load atmospheric absorption coefficient table.

            :return: None
            """

            # Load atmospheric absorption coefficient table
            # Source: verification noise assessment data set of NASA STCA (Berton et al., 2019)
            data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/isa/atmospheric_absorption.csv')
            self.abs_f = data.values[0, 1:]
            self.abs_z = data.values[1:, 0]
            self.abs_data = data.values[1:, 1:]
            
            self.abs_f = interpolate.interp2d(self.abs_f, self.abs_z, self.abs_data, kind='linear')

            return None

        def get_faddeeva_tables(self) -> None:
            """
            
            """

            # Load Faddeeva tables
            data_faddeeva_real = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/propagation/Faddeeva_real_small.csv')
            data_faddeeva_imag = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/propagation/Faddeeva_imag_small.csv')
            self.faddeeva_itau_re = data_faddeeva_real.values[0, 1:]
            self.faddeeva_itau_im = data_faddeeva_real.values[1:, 0]
            self.faddeeva_real = data_faddeeva_real.values[1:,1:]
            self.faddeeva_imag = data_faddeeva_imag.values[1:,1:]

            return None


    class Levels:
        
        def __init__(self) -> None:
            
            Tables.Levels.get_noy_table(self)
            Tables.Levels.get_a_weighting_table(self)

        def get_noy_table(self) -> None:
            """
            Load noy tables for tone-corrected perceived noise level (pnlt) computation.

            :return: None
            """

            # Load noise level computation tabular data
            # Source: ICAO Annex 16 Volume 1 (Edition 8) Table A2-3.
            # self.Noy = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/levels/pnlt_noy_weighting.csv')

            data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/tables/levels/spl_noy.csv')
            self.noy_spl = data.values[1:, 0]
            self.noy_f = data.values[0, 1:]
            self.noy_data = data.values[1:,1:]

            self.noy_f = interpolate.interp2d(self.noy_f, self.noy_spl, self.noy_data, kind='linear')

            return None

        def get_a_weighting_table(self) -> None:
            """
            Load A-weighting coefficients for A-SPL computation.

            :return: None
            """

            self.aw_f = jnp.array([10., 20., 40., 50., 63., 80., 100., 125., 160., 200., 250., 315., 400., 500., 630., 800., 1000., 1250., 1600., 2000., 2500., 3150., 4000., 5000., 6300., 8000., 10000., 12500., 16000., 20000.])
            self.aw_db = jnp.array([-70.4, -50.4, -34.5, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0., 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.7, -9.3])

            return None


    

    


    
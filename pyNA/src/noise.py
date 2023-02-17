import pandas as pd
from pyNA.src.utils.compute_frequency_bands import compute_frequency_bands
from pyNA.src.utils.compute_frequency_subbands import compute_frequency_subbands


class Noise:
    
    def __init__(self, settings, engine, airframe, noise_data) -> None:
        
        self.engine = engine
        self.airframe = airframe
        self.noise_data = noise_data

        self.path = pd.DataFrame()

        self.f = compute_frequency_bands(n_frequency_bands=settings['n_frequency_bands'])
        self.f_sb = compute_frequency_subbands(f=self.f, n_frequency_subbands=settings['n_frequency_subbands'])

    def initialize(self, settings):
        # Load engine time series from file
        self.engine.load_timeseries_csv(file_name=settings['engine_file_name'])

        # Load path time series from file
        self.path.load_timeseries_csv(file_name=settings['path_file_name'])

    def load_path_timeseries_csv(self, file_name:str):
        self.path = pd.read_csv(file_name)

    def compute_noise_level():
        pass

    def compute_epnl_table():
        pass

    def compute_noise_contours():
        pass

    def compute_noise_contour_area():
        pass

    def plot_noise_level():
        pass

    def plot_noise_contours():
        pass

    def plot_noise_source_distribution():
        pass
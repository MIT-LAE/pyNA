import pandas as pd
from pyNA.src.noise_model.tables import Tables
from pyNA.src.noise_model.python.utils.compute_frequency_bands import compute_frequency_bands
from pyNA.src.noise_model.python.utils.compute_frequency_subbands import compute_frequency_subbands


class Noise:
    
    def __init__(self, settings) -> None:
        
        self.tables = Tables(settings=settings)

        self.f = compute_frequency_bands(n_frequency_bands=settings['n_frequency_bands'])
        self.f_sb = compute_frequency_subbands(f=self.f, n_frequency_subbands=settings['n_frequency_subbands'])

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
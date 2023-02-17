import pandas as pd
import numpy as np
import pdb


class Engine():

    def __init__(self, pyna_directory, ac_name, case_name, output_directory_name, engine_timeseries_name, engine_deck_name) -> None:
        
        # Settings 
        self.pyna_directory = pyna_directory
        self.ac_name = ac_name
        self.case_name = case_name
        self.output_directory_name = output_directory_name
        self.engine_timeseries_name = engine_timeseries_name
        self.engine_deck_name = engine_deck_name

        # Instantiate 
        self.timeseries = pd.DataFrame()
        self.deck = dict()
        self.deck_variables = dict()
        

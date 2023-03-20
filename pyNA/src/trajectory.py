import matplotlib.pyplot as plt
import dymos as dm
import pandas as pd
import numpy as np
import openmdao.api as om
import pdb
import pyNA
from pyNA.src.aircraft import Aircraft


class Trajectory:
    
    """
    
    Attributes
    ----------
    problem : om.Problem
        _
    settings : dict
        _
    vars : list
        _
    var_units : dict
        _
    model : dm.Trajectory
        _
    n_t : int
        _
        
    """

    def __init__(self, problem: om.Problem, settings: dict, aircraft: Aircraft) -> None:

        self.problem = problem
        self.settings = settings
        self.aircraft = aircraft
        self.vars = list()
        self.var_units = dict()
        self.model = dm.Trajectory()

    def connect(self, model=dm.Trajectory()) -> None:
        
        """
        
        Parameters
        ----------
        trajectory_mode : str
            'model' or 'time_history'
        
        path : dm.Trajectory()
            Trajectory model

        """

        self.model = model
        self.model.connect_to_model(problem=self.problem, settings=self.settings, aircraft=self.aircraft)
        self.n_t = self.model.n_t

        return None
    
    def set_initial_conditions(self, path_init=None) -> None:
        """
        
        Parameters
        ----------
        problem : 
            _
        settings : dict
            pyna settings
        aircraft : Aircraft
            _
        path_init : 
            _
        
        """

        self.model.set_initial_conditions(problem=self.problem, settings=self.settings, aircraft=self.aircraft, path_init=path_init)

        return None



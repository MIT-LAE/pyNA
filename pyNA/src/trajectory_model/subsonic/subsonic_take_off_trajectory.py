import openmdao.api as om
import dymos as dm
import pyNA
import pdb
from pyNA.src.aircraft import Aircraft


class SubsonicTakeOffTrajectory(dm.Trajectory):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def connect(self, problem: om.Problem, settings: dict, aircraft: Aircraft):

        """
        
        Parameters
        ----------
        problem : om.Problem
            _
        settings : dict
            pyna settings
        aircraft : Aircraft
            _

        """
                
        pass

    def set_initial_conditions(self, problem: om.Problem, settings: dict, aircraft:Aircraft, path_init=None):

        """
        
        Parameters
        ----------
        problem : om.Problem
            _
        settings : dict
            pyna settings
        aircraft : Aircraft
            _
        path_init : 
            _
             
        """

        pass

from pyNA.src.trajectory import Trajectory
from pyNA.src.aircraft import Aircraft
import pdb

class pyna:

    def __init__(self, settings, trajectory_mode='data') -> None:
        
        """
        
        :param trajectory_mode: data or model
        :type trajectory_mode: str
        """

        self.settings = settings

        self.aircraft = Aircraft(settings=settings)

        self.trajectory = Trajectory(settings=settings, mode=trajectory_mode)

        

        # if trajectory_mode == 'model':
            # self.aircraft.get_aerodynamics_deck(settings=settings)
            # self.aircraft.engine.get_performance_deck(settings=settings)

from pyNA.src.trajectory import Trajectory

class pyna:

    def __init__(self, settings, trajectory_mode='data') -> None:
        
        self.settings = settings

        self.trajectory = Trajectory(settings=settings, mode=trajectory_mode)
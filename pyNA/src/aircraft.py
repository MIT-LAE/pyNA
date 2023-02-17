import pandas as pd
from pyNA.src.airframe import Airframe
from pyNA.src.engine import Engine


class Aircraft:
    
    def __init__(self) -> None:
        
        self.engine = Engine()
        self.airframe = Airframe()
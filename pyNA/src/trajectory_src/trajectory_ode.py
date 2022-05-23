import pdb
import openmdao.api as om
from pyNA.src.aircraft import Aircraft
from pyNA.src.engine import Engine
from pyNA.src.settings import Settings
from pyNA.src.trajectory_src.atmosphere import Atmosphere
from pyNA.src.trajectory_src.flight_dynamics import FlightDynamics
from pyNA.src.trajectory_src.clcd import CLCD
from pyNA.src.trajectory_src.ts_limit import TSLimit
from pyNA.src.trajectory_src.aerodynamics import Aerodynamics
from pyNA.src.trajectory_src.propulsion import Propulsion
from pyNA.src.emissions import Emissions


class TrajectoryODE(om.Group):

    """
    Noise model group. The noise model group connects the following components:

    * Atmosphere:           compute atmospheric properties along the trajectory
    * CLCD:                 compute aircraft lift and drag coefficient along the trajectory
    * Propulsion:           compute engine parameters along the trajectory
    * FlightDynamics:       compute flight dynamics equations of motion along the trajectory
    * Emissions:            compute NOx emissions along the trajectory

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('phase', types=str)
        self.options.declare('ac', types=Aircraft)
        self.options.declare('engine', types=Engine)
        self.options.declare('settings', types=Settings)
        self.options.declare('objective', str)

    def setup(self):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        ac = self.options['ac']
        engine = self.options['engine']
        settings = self.options['settings']

        # Atmosphere module
        self.add_subsystem(name='atmosphere',
                           subsys=Atmosphere(num_nodes=nn, settings=settings),
                           promotes_inputs=['z'],
                           promotes_outputs=['p_0', 'rho_0', 'drho_0_dz', 'T_0', 'c_0', 'mu_0', 'I_0'])

        # Aerodynamics module
        self.add_subsystem(name='clcd',
                           subsys=CLCD(vec_size=nn, extrapolate=True, method='3D-lagrange3', ac=ac),
                           promotes_inputs=['alpha', 'theta_flaps', 'theta_slats'],
                           promotes_outputs=[])

        self.add_subsystem(name='aerodynamics', 
                           subsys=Aerodynamics(num_nodes=nn, ac=ac, phase=phase_name),
                           promotes_inputs=['v', 'c_0', 'rho_0'],
                           promotes_outputs=[])
        self.connect('clcd.c_l', 'aerodynamics.c_l')
        self.connect('clcd.c_d', 'aerodynamics.c_d')

        # Propulsion module
        self.add_subsystem(name='propulsion',
                        subsys=Propulsion(vec_size=nn, extrapolate=True, method='3D-lagrange3', settings=settings, engine=engine),
                        promotes_inputs=['z'],
                        promotes_outputs=[])
        self.connect('aerodynamics.M_0', 'propulsion.M_0')
        
        # flight dynamics module
        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightDynamics(num_nodes=nn, settings=settings, phase=phase_name, ac=ac, objective=self.options['objective']),
                           promotes_inputs=['x', 'z', 'v', 'alpha', 'gamma', 'rho_0', 'c_0', 'drho_0_dz'],
                           promotes_outputs=[])
        self.connect('propulsion.F_n', 'flight_dynamics.F_n')
        self.connect('aerodynamics.L', 'flight_dynamics.L')
        self.connect('aerodynamics.D', 'flight_dynamics.D')
        self.connect('clcd.c_l_max', 'flight_dynamics.c_l_max')

        # Emissions module
        self.add_subsystem(name='emissions',
                           subsys=Emissions(num_nodes=nn, settings=settings),
                           promotes_inputs=[],
                           promotes_outputs=[])
        self.connect('propulsion.W_f', 'emissions.W_f')
        self.connect('propulsion.Tti_c', 'emissions.Tti_c')
        self.connect('propulsion.Pti_c', 'emissions.Pti_c')

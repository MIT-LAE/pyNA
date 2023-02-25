import pdb
import openmdao.api as om
from pyNA.src.engine import Engine
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_model.atmosphere import Atmosphere
from pyNA.src.trajectory_model.flight_dynamics import FlightDynamics
from pyNA.src.trajectory_model.aerodynamics import Aerodynamics
from pyNA.src.trajectory_model.propulsion import Propulsion
from pyNA.src.trajectory_model.emissions import Emissions


class TakeOffPhaseODE(om.Group):

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
        self.options.declare('settings', types=dict)
        self.options.declare('aircraft', types=Aircraft)
        self.options.declare('objective', types=str)

    def setup(self):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        settings = self.options['settings']
        aircraft = self.options['aircraft']
        
        # Atmosphere module
        self.add_subsystem(name='atmosphere',
                            subsys=Atmosphere(num_nodes=nn, atmosphere_dT=settings['atmosphere_dT'], mode=settings['atmosphere_mode']),
                            promotes_inputs=['z'],
                            promotes_outputs=['p_0', 'rho_0', 'drho_0_dz', 'T_0', 'c_0', 'mu_0', 'I_0'])

        # Aerodynamics module
        self.add_subsystem(name='aerodynamics',
                        subsys=Aerodynamics(vec_size=nn, extrapolate=True, method='3D-lagrange3', aircraft=aircraft),
                        promotes_inputs=['alpha', 'theta_flaps', 'theta_slats'],
                        promotes_outputs=[])

        # Propulsion module
        self.add_subsystem(name='propulsion',
                        subsys=Propulsion(vec_size=nn, extrapolate=True, method='3D-lagrange3', engine=aircraft.engine, atmosphere_mode=settings['atmosphere_mode']),
                        promotes_inputs=['z'],
                        promotes_outputs=[])
        self.connect('flight_dynamics.M_0', 'propulsion.M_0')
        
        # Flight dynamics module
        self.add_subsystem(name='flight_dynamics',
                        subsys=FlightDynamics(num_nodes=nn, phase=phase_name, settings=settings, aircraft=aircraft, objective=self.options['objective']),
                        promotes_inputs=['x', 'z', 'v', 'alpha', 'gamma', 'c_0', 'rho_0'],
                        promotes_outputs=[])
        self.connect('propulsion.F_n', 'flight_dynamics.F_n')
        self.connect('aerodynamics.c_l', 'flight_dynamics.c_l')
        self.connect('aerodynamics.c_d', 'flight_dynamics.c_d')
        self.connect('aerodynamics.c_l_max', 'flight_dynamics.c_l_max')

        # Emissions module
        self.add_subsystem(name='emissions',
                           subsys=Emissions(num_nodes=nn),
                           promotes_inputs=[],
                           promotes_outputs=[])
        self.connect('propulsion.W_f', 'emissions.W_f')
        self.connect('propulsion.core_Tt_i', 'emissions.core_Tt_i')
        self.connect('propulsion.core_Pt_i', 'emissions.core_Pt_i')

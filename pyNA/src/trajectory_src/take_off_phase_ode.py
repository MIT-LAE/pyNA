import pdb
import openmdao.api as om
from pyNA.src.airframe import Airframe
from pyNA.src.engine import Engine

from pyNA.src.trajectory_src.atmosphere import Atmosphere
from pyNA.src.trajectory_src.flight_dynamics import FlightDynamics
from pyNA.src.trajectory_src.clcd import CLCD
from pyNA.src.trajectory_src.aerodynamics import Aerodynamics
from pyNA.src.trajectory_src.propulsion import Propulsion
from pyNA.src.trajectory_src.emissions import Emissions


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
        self.options.declare('airframe', types=Airframe)
        self.options.declare('engine', types=Engine)
        self.options.declare('sealevel_atmosphere', types=dict)
        self.options.declare('atmosphere_dT', types=float)
        self.options.declare('atmosphere_type', types=str)
        self.options.declare('objective', types=str)
        self.options.declare('case_name', types=str)
        self.options.declare('output_directory_name', types=str)

    def setup(self):
        # Load options
        nn = self.options['num_nodes']
        phase_name = self.options['phase']
        
        # Atmosphere module
        if self.options['atmosphere_type'] == 'stratified':
            self.add_subsystem(name='atmosphere',
                               subsys=Atmosphere(num_nodes=nn, sealevel_atmosphere=self.options['sealevel_atmosphere'], atmosphere_dT=self.options['atmosphere_dT']),
                               promotes_inputs=['z'],
                               promotes_outputs=['p_0', 'rho_0', 'drho_0_dz', 'T_0', 'c_0', 'mu_0', 'I_0'])

        # Aerodynamics module
        self.add_subsystem(name='clcd',
                        subsys=CLCD(vec_size=nn, extrapolate=True, method='3D-lagrange3', airframe=self.options['airframe']),
                        promotes_inputs=['alpha', 'theta_flaps', 'theta_slats'],
                        promotes_outputs=[])

        if self.options['atmosphere_type'] == 'stratified':
            self.add_subsystem(name='aerodynamics', 
                            subsys=Aerodynamics(num_nodes=nn, phase=phase_name, airframe=self.options['airframe'], atmosphere_type=self.options['atmosphere_type'], sealevel_atmosphere=self.options['sealevel_atmosphere']),
                            promotes_inputs=['v', 'c_0', 'rho_0'],
                            promotes_outputs=[])
        else:
            self.add_subsystem(name='aerodynamics', 
                            subsys=Aerodynamics(num_nodes=nn, phase=phase_name, airframe=self.options['airframe'], atmosphere_type=self.options['atmosphere_type'], sealevel_atmosphere=self.options['sealevel_atmosphere']),
                            promotes_inputs=['v'],
                            promotes_outputs=[])
        self.connect('clcd.c_l', 'aerodynamics.c_l')
        self.connect('clcd.c_d', 'aerodynamics.c_d')

        # Propulsion module
        if self.options['atmosphere_type'] == 'stratified':
            self.add_subsystem(name='propulsion',
                            subsys=Propulsion(vec_size=nn, extrapolate=True, method='3D-lagrange3', engine=self.options['engine'], atmosphere_type=self.options['atmosphere_type']),
                            promotes_inputs=['z'],
                            promotes_outputs=[])
        else:
            self.add_subsystem(name='propulsion',
                            subsys=Propulsion(vec_size=nn, extrapolate=True, method='slinear', engine=self.options['engine'], atmosphere_type=self.options['atmosphere_type']),
                            promotes_inputs=[],
                            promotes_outputs=[])
        self.connect('aerodynamics.M_0', 'propulsion.M_0')
        
        # Flight dynamics module
        self.add_subsystem(name='flight_dynamics',
                        subsys=FlightDynamics(num_nodes=nn, phase=phase_name, airframe=self.options['airframe'], sealevel_atmosphere=self.options['sealevel_atmosphere'], objective=self.options['objective'], case_name=self.options['case_name'], output_directory_name=self.options['output_directory_name']),
                        promotes_inputs=['x', 'z', 'v', 'alpha', 'gamma'],
                        promotes_outputs=[])
        self.connect('propulsion.F_n', 'flight_dynamics.F_n')
        self.connect('aerodynamics.L', 'flight_dynamics.L')
        self.connect('aerodynamics.D', 'flight_dynamics.D')
        self.connect('clcd.c_l_max', 'flight_dynamics.c_l_max')

        # Emissions module
        self.add_subsystem(name='emissions',
                           subsys=Emissions(num_nodes=nn),
                           promotes_inputs=[],
                           promotes_outputs=[])
        self.connect('propulsion.W_f', 'emissions.W_f')
        self.connect('propulsion.Tti_c', 'emissions.Tti_c')
        self.connect('propulsion.Pti_c', 'emissions.Pti_c')

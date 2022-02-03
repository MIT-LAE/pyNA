.. _settings:

settings
========

class description
-----------------

.. automodule:: settings
	:members:
	:undoc-members:
	:show-inheritance:

sample settings file
--------------------

.. code::

	# Initialize settings
	pyna_settings = dict()

	# Directories and file names
	pyna_settings['case_name'] = 'NASA STCA Standard'                    # Case name [-]
	pyna_settings['case_directory'] = 'pyNA/cases/NASA STCA Standard'    # Case directory [-]
	pyna_settings['trajectory_file_name'] = 'Trajectory_to.csv'          # Name of the trajectory in the setup/traj folder [-]
	pyna_settings['engine_file_name'] = 'Inputs_to.csv'                  # File name of the take-off engine inputs [-]
	pyna_settings['aero_file_name'] = 'aerodeck.csv'                     # File nameme of the aerodynamics deck [-]
	pyna_settings['output_file_name'] = 'Trajectory_STCA.sql'            # Name of the output .sql file [-]
	pyna_settings['ac_name'] = 'stca'                                    # Aircraft name [-]

	# Mode
	pyna_settings['language'] = 'python'            # Language to use to solve components (julia/python) [-]
	pyna_settings['save_results'] = True            # Flag to save results [-]

	# Noise settings
	pyna_settings['fan_inlet'] = False              # Enable fan inlet noise source (True/False)
	pyna_settings['fan_discharge'] = False          # Enable fan discharge noise source (True/False)
	pyna_settings['core'] = False                   # Enable core noise source (True/False)
	pyna_settings['jet_mixing'] = True              # Enable jet mixing noise source (True/False)
	pyna_settings['jet_shock'] = False              # Enable jet shock noise source (True/False)
	pyna_settings['airframe'] = False               # Enable airframe noise source (True/False)
	pyna_settings['all_sources'] = False            # Enable all noise sources (True/False)
	pyna_settings['observer'] = 'lateral'           # Observers to analyze [-] ('flyover','lateral','approach')
	pyna_settings['method_core_turb'] = 'GE'        # Method to account for turbine transmission in the combustor ('GE', 'PW') [-]
	pyna_settings['fan_BB_method'] = 'geae'         # Method BB (original / allied_signal / geae / kresja) [-]
	pyna_settings['fan_RS_method'] = 'allied_signal'# Method RS (original / allied_signal / geae / kresja) [-]
	pyna_settings['fan_igv'] = False                # Fan inlet guide vanes (True/False)
	pyna_settings['fan_id'] = False                 # Fan inlet distortions (True/False)
	pyna_settings['ge_flight_cleanup'] = 'takeoff'  # GE flight cleanup switch (none / takeoff / approach) [-]

	# Flags
	pyna_settings['absorption'] = True              # Flag for atmospheric absorption
	pyna_settings['groundeffects'] = True           # Flag for ground effects
	pyna_settings['suppression'] = True             # Flag for suppression of engine modules
	pyna_settings['shielding'] = False              # Flag for shielding effects (not implemented yet)
	pyna_settings['hsr_calibration'] = True         # Flag for HSR-era airframe calibration
	pyna_settings['validation'] = True              # Flag for validation with NASA STCA noise model
	pyna_settings['bandshare'] = False              # Flag to plot PNLT
	pyna_settings['TCF800'] = True                  # Flag for tone penalty addition to PNLT metric; allows any tone below 800Hz to be ignored
	pyna_settings['combination_tones'] = False      # Flag for combination tones int he fan noise model

	# Constants
	pyna_settings['N_e'] = 3                        # Number of engines[-]
	pyna_settings['N_shock'] = 8                    # Number of shocks in supersonic jet [-]
	pyna_settings['dT'] = 10.0169                   # dT standard atmosphere [K]
	pyna_settings['sigma'] = 291.0 * 515.379        # Specific flow resistance of ground [kg/s m3]
	pyna_settings['a_coh'] = 0.01                   # Incoherence constant [-]
	pyna_settings['N_f'] = 24                       # Number of discrete 1/3 octave frequency bands [-]
	pyna_settings['N_b'] = 5                        # Number of bands (propagation) [-]
	pyna_settings['n_altitude_absorption'] = 5      # Number of integration steps in atmospheric propagation [-]
	pyna_settings['A_e'] = 10.334 * (0.3048 ** 2)   # Engine reference area [m2]
	pyna_settings['dt_epnl'] = 0.5                  # Time step of to calculate EPNL from interpolated PNLT data [s]
	pyna_settings['n_harmonics'] = 10               # Number of harmonics to be considered in tones [-]
	pyna_settings['r_0'] = 0.3048                   # Distance source observer in source mode [m]
	pyna_settings['p_ref'] = 2e-5                   # Reference pressure [Pa]

	# Trajectory settings
	if pyna_settings['case_name'] in ['case_1', 'stca', 'a10']:
	    pyna_settings['trajectory_mode'] = 'compute'                    # Mode for trajectory calculations [-] ('load' / 'compute')
	    pyna_settings['trajectory_optimize_noise'] = False              # Flag to noise-optimize the trajectory [-]
	elif pyna_settings['case_name'] == 'NASA STCA Standard':    
	    pyna_settings['trajectory_mode'] = 'load'                       # Mode for trajectory calculations [-] ('load' / 'compute')
	    pyna_settings['trajectory_optimize_noise'] = False              # Flag to noise-optimize the trajectory [-]

	pyna_settings['trajectory_climb_phase_lst'] = ['climb', 'climb2']   # Climb phase names [-]
	pyna_settings['trajectory_groundroll_TS'] = 0.95                    # Groundroll thrust setting [-]
	pyna_settings['trajectory_rotation_TS'] = 0.95                      # Rotation thrust setting [-]
	pyna_settings['trajectory_climb_alt_lst'] = [0, 210, 1500.]         # Climb phases altitude list [-]
	pyna_settings['trajectory_climb_TS_lst'] = [0.95, 0.65*0.95]        # Climb phases thrust setting list [-]

	pyna_settings['trajectory_num_segments'] = 5                        # Trajectory number of segments [-]
	pyna_settings['trajectory_transcription_order'] = 3                 # Trajectory dymos transcription ordor [-]
	pyna_settings['trajectory_max_iter'] = 250                          # Maximum number of iterations for trajectory computations [-]


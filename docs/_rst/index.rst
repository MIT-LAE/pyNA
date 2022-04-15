.. pyNA documentation master file, created by
   sphinx-quickstart on Tue Feb  9 12:26:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Welcome to pyNA's documentation!
.. ================================

.. image:: ./_images/logo.jpg
   :width: 700

**pyNA** is the *python Noise Assessment* tool to assess the aircraft noise during take-off operations. The tool estimates the mean-square acoustic pressure level for different engine noise sources, propagates the source noise to an observer on the ground and calculates the noise levels as defined in ICAO Annex 16 Environmental Protection, Volume I: Aircraft Noise.

Welcome to pyNA's documentation!
--------------------------------

Installation
^^^^^^^^^^^^

Python version of pyNA
""""""""""""""""""""""

Get pyNA from a cloned repository from `Github <https://github.com/MIT-LAE/pyNA>`_:

   git clone git@github.com:MIT-LAE/pyNA.git

Use pip to install pyNA:

   pip install -e .

The python version is default in pyNA. To enable this mode, set the python environment variable ``pyna_language`` equal to 'python':

.. code-block::
   
   import os
   os.environ['pyna_language'] = 'python'

Julia version of pyNA
"""""""""""""""""""""

To enable fast computation of sensitivities of acoustic objective functions in pyNA, an installation of Julia is required since the modules *geometry, source, propagation and levels* are using Julia's ForwardDiff. Install `Julia <https://julialang.org>`_. To enable this mode, set the python environment variable ``pyna_language`` equal to 'julia':

.. code-block::
   
   import os
   os.environ['pyna_language'] = 'julia'


Citation
^^^^^^^^^^^^
If you utilize pyNA in your work, please reference it using the following citation:

.. code-block:: latex

   @unpublished{Voet2022,
     author = {Laurens J. A. Voet and Prashanth Prakash and Raymond L. Speth and Jayant S. Sabnis and Choon S. Tan and Steven R. H. Barret},
     title = {Sensitivities of aircraft acoustic metrics to engine design and control variables for multi-disciplinary optimization},
     journal = {AIAA Journal (manuscript under review)},
     year = {2022},
   }



Indices and tables
^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Main

   how_to_use
   methods
   openmdao
   partials

.. toctree::
   :maxdepth: 1
   :caption:  Examples

   example_noise_time_series
   example_noise_contours
   example_noise_epnl_table
   example_noise_source_distribution

.. toctree::
   :maxdepth: 1
   :caption: pyNA

   aircraft
   data
   emissions
   engine
   noise
   settings
   trajectory
   pyNA

.. toctree::
   :maxdepth: 1
   :caption: Noise

   airframe
   core
   epnl
   fan
   geometry
   ground_reflections
   ioaspl
   ipnlt
   jet
   lateral_attenuation
   levels_int
   levels
   noise
   noise_model
   normalization_engine_variables
   oaspl
   pnlt
   propagation
   shielding
   source
   spl
   split_subbands
   
.. toctree::
   :maxdepth: 1
   :caption: Trajectory

   aerodynamics
   atmosphere
   clcd
   flight_dynamics
   mux
   propulsion
   trajectory
   trajectory_ode

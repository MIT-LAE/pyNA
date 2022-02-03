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

.. toctree::
   :maxdepth: 4
   :caption: Theory

   how_to_use
   methods
   openmdao
   partials

.. toctree::
   :maxdepth: 4
   :caption: Support

   installation
   help!

.. toctree::
   :maxdepth: 4
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
   :maxdepth: 4
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
   :maxdepth: 4
   :caption: Trajectory

   aerodynamics
   atmosphere
   clcd
   flight_dynamics
   mux
   propulsion
   trajectory
   trajectory_ode


Installation
------------
Look at the :ref:`installation` page for detailed instructions to get pyNA working on your machine. 

Getting started
---------------
Look at the *examples* section for examples of a settings file, a run file and a post-processing file.

Citation
--------
If you utilize pyNA in your work, please reference it using the following citation:

.. code-block:: latex

   @unpublished{Voet2021,
      author = {Laurens. J. A. Voet, Raymond L. Speth, Jayant S. Sabnis, Choon S. Tan and Steven R. H. Barrett},
      title = {Development of optimal control framework to design variable noise reduction systems for take-off operations of civil supersonic transport aircraft (work in progress).},
      year = {2021},
   }

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

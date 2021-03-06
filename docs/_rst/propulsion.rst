propulsion
==========

.. automodule:: propulsion
   :members:
   :undoc-members:
   :show-inheritance:


The thermodynamic engine cycle module is developed using the numerical propulsion system simulation (NPSS) software. A look-up table is implemented of the engine parameters as a function of flight Mach number, :math:`M_0 \in [0,0.5]`, flight altitude, :math:`z\in [0, 5000]m`, and engine thrust-setting, :math:`TS \in [30, 105] \%`. The thrust-setting, TS, at a particular Mach number and altitude is defined as a fraction of the maximum thermodynamic available thrust at these flight conditions. The maximum available thrust is defined as the thrust obtained when the engine is operated at 100\% fan corrected speed (N\textsubscript{1c2}).
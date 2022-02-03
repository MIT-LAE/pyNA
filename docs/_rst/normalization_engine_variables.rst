normalization\_engine\_variables
================================

class description
-----------------

.. automodule:: normalization_engine_variables
   :members:
   :undoc-members:
   :show-inheritance:

theory
------

As described by Zorumski, the engine variables are normalized before being fed to the noise module. After normalization, the variables are denoted with superscript \*. Engine temperatures, velocities and densities are normalized by their ambient counterparts; dimensions are normalized by the corresponding power of a reference area, :math:`A_e`; mass flows are normalized by the product :math:`\rho_0 c_0 A_e` and the fan rotational speed is normalized by :math:`60\frac{c_0}{d_fan}`.
aerodynamics
============

class description
-----------------

.. automodule:: aerodynamics
   :members:
   :undoc-members:
   :show-inheritance:

theory
------

The aircraft aerodynamic lift coefficient, :math:`C_L`, and drag coefficient, :math:`C_D`, are provided to the module in terms of a look-up table as a function of the wing angle of attack, :math:`\alpha`, and the flap and slat deflection angle, :math:`\theta_{flap}` and :math:`\theta_{slat}`. The aircraft lift and drag forces are computed using the wing surface area, :math:`S`, and the dynamic pressure, :math:`q=\frac{1}{2}\rho v^2`, as follows: 

.. math::

    \begin{array}{lp{2cm}l}
    L = C_L(\alpha, \theta_{flap}, \theta_{slat}) q S && D = C_D(\alpha, \theta_{flap} \theta_{slat}) q S \\
    \end{array}

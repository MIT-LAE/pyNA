atmosphere
==========

class description
-----------------

.. automodule:: atmosphere
   :members:
   :undoc-members:
   :show-inheritance:

theory
------

The 1976 US Standard Atmospheric (USSA) model computes the ambient temperature, :math:`T_0`, pressure, :math:`p_0`, density, :math:`\rho_0`, speed of sound, :math:`c_0`, dynamic viscosity, $\mu_0$, and characteristic impedance, :math:`I_0`, at altitude, :math:`z`, given the sea level conditions (referenced by subscript sl). A temperature deviation from the USSA model is implemented using :math:`\Delta T_{USSA}`. The ratio of specific heats, the gravitational constant, the air gas constant and the atmospheric lapse rate are given by $\gamma$, :math:`g`, :math:`R`, and :math:`\lambda`, respectively. 

.. math::

    \begin{array}{lp{2cm}l}
    T_{USSA} = T_{sl} - \lambda z && c_0 = \sqrt{\gamma R T_0}\\
    T_0 = T_{sl} - \lambda z + \Delta T_{USSA} && \mu_0 = \mu_{sl} \left(1.38313 \left[\frac{T_0}{T_{sl}}\right]^{1.5} \right) \bigg/ \left(\frac{T_0}{T_{sl}} + 0.38313 \right) \\   % Sutherlands equation\\
    p_0 = p_{sl} \left(\frac{T_{USSA}}{T_{sl}}\right)^{\frac{g}{\lambda R}} && I_0 = \rho_{sl} c_{sl} \left(\frac{p_0}{p_{sl}}\right) \sqrt{\frac{T_{sl}}{T_0}}\\
    \rho_0 = \frac{p_0}{RT_0} &&\\
    \end{array}
core
====

class description
-----------------

.. automodule:: core
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members:

theory
------

The core noise source is governed by the unsteady heat addition inside the combustor and is attenuated as it passes through the downstream turbine components. The Emmerling method is used to estimate the mean-square acoustic pressure of the combustor noise source:

.. math::

    <p^2(r_s^*)>^* = 10\log_{10} \left[ \frac{\Pi^* A^*}{4\pi(r_s^*)^2 p_{\textrm{ref}}^2}\frac{D(\theta) F(f/f_p)}{(1-M_0 \cos \theta )^4} \right]

where:

.. math::

    \Pi^* = (8.85\cdot 10^{-7}) \frac{\dot{m}_i^*}{A^*}\left(\frac{T_{t,j}^*-T_{t,i}^*}{T_{t,i}^*}\right)^2(p_{t,i})^2 g_{tt}

The peak frequency, :math:`f_p`, in the spectral distribution function, :math:`F`, is given by :math:`f_p = \frac{400}{1-M_0 cos(\theta)}`. The directivity function function, :math:`D(\theta)`, and the spectral distribution function, :math:`F(f/f_p)`, are tabulated online.
Two turbine transmission loss functions can be used, namely the GE turbine transmission loss function :math:`g_{{tt}_{GE}} =  (\Delta T_{turb., des}^*)^{-4}` and the Pratt \& Whitney turbine transmission loss function :math:`g_{{tt}_{PW}} = \frac{0.8\zeta}{(1+\zeta)^2}`, with :math:`\zeta = \frac{\rho_{turb.exit}c_{turb.exit}}{\rho_{turb.in}c_{turb.in}}`.

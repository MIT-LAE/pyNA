split subbands
==============

class description
-----------------

.. automodule:: split_subbands
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members:

theory
------

The atmospheric absorption and ground absorption and reflections are applied to a sub-band frequency spectrum. Each frequency, :math:`f_i`, within the  one-third octave frequency spectrum (i.e. :math:`i \in [1, N_f]`) is divided into :math:`N_b = 2m+1` sub-bands, where :math:`m` is a strictly positive integer. The ratio of sub-band center frequencies is:

.. math::
	
	\frac{f_{sb, j+1}}{f_{sb, j}} = w = 2^{1/(3N_b)}

where :math:`j = (i-1)N_b + h` is the index of the sub-band center frequency and :math:`h \in [1,N_b]`. Thus, the j-th sub-band center frequency is computed from the i-th original frequency using: 

.. math::

    f_j = w^{h-m-1} f_i \quad \textrm{with} h \in [1, N_b] \ \textrm{and}\ i \in [1, N_f]

The mean-square acoustic pressure of each original one-third frequency band, :math:`<p^2>^*_i`, is also divided into sub-bands. Firstly, the slopes of the mean-square acoustic pressure in the lower (:math:`u`) and upper (:math:`v`) half of the band are computed using:

.. math::

    \begin{cases}
    u_i = \frac{<p^2>^*[i]}{<p^2>^*[i-1]}; v = \frac{<p^2>^*[i+1]}{<p^2>^*[i]}\quad & \textrm{if} \quad i \in [2, N_f]\\
    u_i = v_i = \frac{<p^2>^*[1]}{<p^2>^*[0]} \quad & \textrm{if} \quad i = 1 \\
    u_i = v_i = \frac{<p^2>^*[N_f]}{<p^2>^*[N_f-1]} \quad & \textrm{if} \quad i = N_f \\
    \end{cases}

The sub-band adjusting factor, :math:`A_i`, is computed using:

.. math::

    A_i = 1 + \sum_{h = 1}^{m} \left( u_i^{(h-m-1)/N_b} + v_i^{1/N_b} \right)

Finally, the mean-square acoustic pressure of each sub-band frequency is given by: 

.. math::

    <p^2>^*_j = 
    \begin{cases}
    (<p^2>^*_i / A_i)u_i^{h-m-1} \quad & \textrm{if} \quad h \in [1, m] \\ 
    (<p^2>^*_i / A_i) \quad & \textrm{if} \quad h = m + 1\\ 
    (<p^2>^*_i / A_i)v_i^{h-m-1} \quad & \textrm{if} \quad h \in [m+2, N_b]\\ 
    \end{cases}

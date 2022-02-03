.. _openmdao_use:

openmdao
========

pyNA is developed in the `openMDAO <https://openmdao.org>`_ framework. On this page, you'll learn what you need to know about openMDAO to be able to interact with pyNA. 

creating an openMDAO problem
----------------------------

When calling ``pyna.create_problem()``, an openMDAO problem, called *pyna.prob* is created with the *NoiseModel* group. The *NoiseModel* group hosts the components that are used to compute noise of the trajectory. 

* The *Geometry* component
* The *Source* component
* The *Propagation* component
* The *Levels* component

A recorder can be added to an openMDAO to save variables in the model after a run. More information about openMDAO recorders can be found `here <http://openmdao.org/twodocs/versions/latest/features/recording/saving_data.html?highlight=case%20recorder>`_. To enable the recording of the results, set ``settings["save_results"] = True``. The output file can be named using ``settings["output_file_name"]``. This file will be saved in the output folder of the ``settings["case_directory"]`` directory.

The command ``prob.setup()`` sets up the model without running it. It connects all the inputs and outputs of all the components.  

The command ``prob.run_model()`` runs the openMDAO problem.  

N2 diagram
----------

A handy feature of an openMDAO problem is the input-output or N2 diagram. The structure of the openMDAO problem is nicely illustrated using such a diagram. An example can be seen in the :ref:`model_structure` page.

accessing variables
-------------------
To access the variables, you can use the command ``prob.get_val("...")``, where ``...`` is the name of the variable. You can use the N2 diagram to see the promoted name of the variables inside pyNA. 

*For example*: the sound-pressure level (SPL) can be accessed by calling ``prob.get_val('levels.SPL')``.

.. image:: ./_images/access_promoted_name.png
   :width: 1000




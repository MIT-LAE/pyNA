.. model_structure:

partials
========

pyNA includes the capability to compute the partial derivatives of the different noise modules with respect to their inputs. 


The partial derivatives of each of the modules are computed using the Julia automatic differentation (AD) package JuliaDiff `JuliaDiff <https://juliadiff.org>`_. To enable this mode, the value ``settings["language"]`` needs to be set to ``"julia"``.






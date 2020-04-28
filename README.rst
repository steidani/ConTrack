|docs| |pipelines|

###############################################
ConTrack - Contour Tracking: A Package to track circulation anomalies.
###############################################

Based on the atmospheric blocking index by Schwierz et al. [2006].
See also:
- Croci-Maspoli et al. (2007)
- Pfahl et al. (2015)
- Steinfeld and Pfahl (2019)

The PV-Anomaly blocking climatology used in Steinfeld and Pfahl (2019) is publicly available via an ETH Zurich-based web server [http://eraiclim.ethz.ch/, see Sprenger et al. (2017)].

==========
Current Status: In early development!!!!!
==========

==========
What's New
==========

v0.1.0 (20.04.2020): 
-------------------

- Extend functionality: Calculating anomalies from daily long-term climatology.
- ...


Install the development environment
-----------------------------------

Copy locally the latest version from lagranto:

.. code-block:: bash

    git clone git@github.com:steidani/ConTrack.git /path/to/local/contrack
    cd path/to/local/contrack

Prepare the conda environment:

.. code-block:: bash

    module load miniconda3
    conda create -y -q -n contrack_dev python=3.7 pytest
    conda env update -q -f contrack.yml -n contrack_dev



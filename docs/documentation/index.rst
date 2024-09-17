Documentation and Tutorials
===========================

Parcels has several documentation and tutorial Jupyter notebooks and scripts which go through various aspects of Parcels. Static versions of the notebooks are available below via the gallery in the site, with the interactive notebooks being available either completely online at the following `Binder link <https://mybinder.org/v2/gh/OceanParcels/parcels/master?labpath=docs%2Fexamples%2Fparcels_tutorial.ipynb>`_. Following the gallery of notebooks is a list of scripts which provide additional examples to users. You can work with the example notebooks and scripts locally by downloading :download:`parcels_tutorials.zip </_downloads/parcels_tutorials.zip>` and running with your own Parcels installation.

.. warning::
   In v3.1.0 we updated kernels in the tutorials to use ``parcels.ParcelsRandom`` instead of ``from parcels import ParcelsRandom``. Due to our C-conversion code, using ``parcels.ParcelsRandom`` only works with v3.1.0+. When browsing/downloading the tutorials, it's important that you are using the documentation corresponding to the version of Parcels that you have installed. You can find which parcels version you have installed by doing ``import parcels`` followed by ``print(parcels.__version__)``. If you don't want to use the latest version of Parcels, you can browse prior versions of the documentation by using the version switcher in the bottom right of this page.

.. nbgallery::
   :caption: Overview
   :name: tutorial-overview

   ../examples/tutorial_parcels_structure.ipynb
   ../examples/parcels_tutorial.ipynb
   ../examples/tutorial_output.ipynb


.. nbgallery::
   :caption: Setting up FieldSets
   :name: tutorial-fieldsets

   ../examples/documentation_indexing.ipynb
   ../examples/tutorial_nemo_curvilinear.ipynb
   ../examples/tutorial_nemo_3D.ipynb
   ../examples/tutorial_croco_3D.ipynb
   ../examples/tutorial_NestedFields.ipynb
   ../examples/tutorial_timevaryingdepthdimensions.ipynb
   ../examples/tutorial_periodic_boundaries.ipynb
   ../examples/tutorial_interpolation.ipynb
   ../examples/tutorial_unitconverters.ipynb
   ../examples/tutorial_timestamps.ipynb


.. nbgallery::
   :caption: Creating ParticleSets
   :name: tutorial-particlesets

   ../examples/tutorial_jit_vs_scipy.ipynb
   ../examples/tutorial_delaystart.ipynb


.. nbgallery::
   :caption: Writing kernels to be executed on each particle
   :name: tutorial-kernels

   ../examples/tutorial_diffusion.ipynb
   ../examples/tutorial_sampling.ipynb
   ../examples/tutorial_particle_field_interaction.ipynb
   ../examples/tutorial_interaction.ipynb
   ../examples/tutorial_analyticaladvection.ipynb
   ../examples/tutorial_kernelloop.ipynb


.. nbgallery::
   :caption: Other tutorials
   :name: tutorial-other

   ../examples/documentation_MPI.ipynb
   ../examples/documentation_stuck_particles.ipynb
   ../examples/documentation_unstuck_Agrid.ipynb
   ../examples/documentation_LargeRunsOutput.ipynb
   ../examples/documentation_geospatial.ipynb
   ../examples/documentation_advanced_zarr.ipynb


.. nbgallery::
   :caption: Worked examples
   :name: tutorial-examples

   ../examples/tutorial_Argofloats.ipynb
   ../examples/documentation_homepage_animation.ipynb


Python Example Scripts
----------------------


.. toctree::

   additional_examples

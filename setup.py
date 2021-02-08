"""Install Parcels and dependencies."""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='parcels',
      description="""Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author="oceanparcels.org team",
      use_scm_version={'write_to': 'parcels/_version.py'},
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      packages=find_packages(),
      package_data={'parcels': ['include/*',
                                'nodes/*',
                                'examples/*']},
      entry_points={'console_scripts': [
          'parcels_get_examples = parcels.scripts.get_examples:main',
          'parcels_convert_npydir_to_netcdf = parcels.scripts.convert_npydir_to_netcdf:main']}
      )

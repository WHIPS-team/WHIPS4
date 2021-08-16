from setuptools import setup, find_packages

REQUIRES = ['cython','numpy', 'shapely', 'pyproj', 'pytables', 'netCDF4', 'jpeg','pyhdf']


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='WHIPS4',
      version='4.0',
      install_requires = REQUIRES,
      description='Scripts for customized regridding of Level-2 data to Level-3 data',
      long_description=readme(),
      author='Tracey Holloway, Jacob Oberman, Peidong Wang, Eliot Kim',
      author_email='taholloway@wisc.edu',
      packages=['process_sat'],
      scripts=['process_sat/whips.py'],
      url = 'https://nelson.wisc.edu/sage/data-and-models/software.php',
      download_url='https://github.com/WHIPS-team/WHIPS4/archive/refs/tags/v4.0.tar.gz',
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Programming Language :: Python'
      ]
)

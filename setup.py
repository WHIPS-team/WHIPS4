from setuptools import setup, find_packages

REQUIRES = ['numpy', 'shapely', 'pyproj', 'tables', 'netCDF4', 'pyhdf']


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='WHIPS3',
      version='3.1.1',
      install_requires = REQUIRES,
      description='Scripts for customized regridding of Level-2 data to Level-3 data',
      long_description=readme(),
      license = 'MIT License',
      author='Jacob Oberman, Peidong Wang, Tracey Holloway',
      author_email='taholloway@wisc.edu',
      packages=['process_sat'],
      scripts=['process_sat/whips.py'],
      url = 'https://nelson.wisc.edu/sage/data-and-models/software.php',
      download_url='https://github.com/WHIPS-team/WHIPS3/archive/3.1.1.tar.gz',
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Programming Language :: Python'
      ]
)

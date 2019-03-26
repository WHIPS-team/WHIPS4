from setuptools import setup, find_packages

REQUIRES = ['numpy', 'shapely', 'pyproj', 'tables', 'netCDF4', 'pyhdf']

setup(name='WHIPS2',
      version='2.0.0',
      install_requires = REQUIRES,
      description='Scripts for customized regridding of Level-2 data to Level-3 data',
      long_description=open('reST.txt').read(),
      license = 'MIT License',
      author='Jacob Oberman, Keith Maki, Xiaomeng Jin',
      author_email='taholloway@wisc.edu',
      packages=['process_sat'],
      scripts=['process_sat/whips.py'],
      url = 'http://www.sage.wisc.edu/download/WHIPS/WHIPS.html',
      download_url='http://github.com/Joberman/process_sat/downloads',
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Programming Language :: Python'
      ]
)

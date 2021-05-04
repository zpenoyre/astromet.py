# to run: python setup.py install

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
#from distutils.extension import Extension

setup(name='astromet',
      version='1.0',
      description='One and two body astrometry',
      author='Zephyr Penoyre',
      author_email='zephyrpenoyre@gmail.com',
      url='https://github.com/zpenoyre/astromet.py',
      license='GNU GPLv3',
      packages=['astromet'],
      install_requires=['numpy','astropy'],
      )

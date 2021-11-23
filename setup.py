# to run: python setup.py install
try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup
#from distutils.extension import Extension

setup(name="astromet",
      version='1.0.16',
      description='One and two body astrometry',
      author='Zephyr Penoyre',
      author_email='zephyrpenoyre@gmail.com',
      url='https://github.com/zpenoyre/astromet.py',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=['numpy','astropy','scipy'],
      include_package_data=True,
      package_data={'': ['data/*.csv']},
      )

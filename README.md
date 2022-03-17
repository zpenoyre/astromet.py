# astromet.py

A simple python package for generating astrometric tracks of single stars and the center of light of unresolved binary, blended and lensed systems. Includes a close emulation of Gaia's astrometric fitting pipeline.

https://astrometpy.readthedocs.io/en/latest/

**pip install astromet**

Still in development, functional but may occasional bugs or future changes. Get in touch with issues/suggestions.

Requires
- numpy
- astropy
- scipy
- numba (optional\*)
- matplotlib (for notebooks)

\* for more accurate solver of keplers equation at high eccentricity set `astromet.use_backup=False` which uses ([Philcox, Goodman & Slepian 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.6111P/abstract) - [github](https://github.com/oliverphilcox/Keplers-Goat-Herd)) but requires numba to run, otherwise default is to use [Penoyre & Sandford 2019, equation A2](https://arxiv.org/pdf/1803.07078.pdf)

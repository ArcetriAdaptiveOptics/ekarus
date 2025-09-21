# EKARUS
Simulation tools for Ekarus. Python libraries to simulate AO loops with a pyramid WFS, correcting KL modes of the turbulence.

## Installation
Unfortunately, the ekarus library has quite a few dependancies, which you will need to install:
'''bash
pip install cupy
pip install xupy
pip install arte
pip install thin-plate-spline
'''
After pip installing the required dependencies, setup a conda environment:
'''bash
conda create -name ekarus arte cupy xupy numpy matplotlib
'''

## Requirements
- Python 3.11+
- numpy
- matplotlib
- cupy (for GPU acceleration)
- xupy (to switch between GPU/CPU)
- arte (for atmospheric layers generation)
- thin-plate-spline (for influence functions simulation)

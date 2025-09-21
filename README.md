# EKARUS
Simulation tools for Ekarus. Python libraries to simulate AO loops with a pyramid WFS, correcting KL modes of the turbulence.

## Installation
Unfortunately, the ekarus library has quite a few dependancies, which you will need to install.
The first two are required for atmospheric layer generation and  synthecic influence functions generation:
```bash
pip install arte
pip install thin-plate-spline
```

The next two are to deal with GPU acceleration:
```bash
pip install cupy
pip install xupy
```

After pip installing the required dependencies, setup a conda environment:
```bash
conda create -name ekarus arte cupy xupy numpy matplotlib
```

You can now activate the environment:
```bash
conda acivate ekarus
```

Now navigate to the download directory and install the package in development mode:
```bash
cd git/ekarus
pip install -e .
```

## Requirements
- Python 3.11+
- numpy
- matplotlib
- cupy (for GPU acceleration)
- xupy (to switch between GPU/CPU)
- arte (for atmospheric layers generation)
- thin-plate-spline (for influence functions simulation)

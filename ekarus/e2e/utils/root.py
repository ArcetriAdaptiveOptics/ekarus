import os.path as op
from os import makedirs

bpath = op.join(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))), 'simulations')
configpath = op.join(bpath, 'Config')
resultspath = op.join(bpath, 'Results')
datapath = op.join(bpath, 'Data')


for p in [bpath, configpath, resultspath, datapath]:
    if not op.exists(p):
        makedirs(p)

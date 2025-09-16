import os.path as op
from os import makedirs

bpath = op.join(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))), 'simulations')
configpath = op.join(bpath, 'Config')
resultspath = op.join(bpath, 'Results')
datapath = op.join(bpath, 'Data')
alpaopath = op.join(op.dirname(bpath), 'e2e', 'alpao_dms')


for p in [bpath, configpath, resultspath, datapath, alpaopath]:
    if not op.exists(p):
        makedirs(p)

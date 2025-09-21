import os.path as op
from os import makedirs

bpath = op.join(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))), 'simulations')
configpath = op.join(bpath, 'config')
resultspath = op.join(bpath, 'results')
calibpath = op.join(bpath, 'calib')
atmopath = op.join(bpath, 'atmo')
alpaopath = op.join(op.dirname(bpath), 'e2e', 'alpao_dms_data')

for p in [bpath, configpath, resultspath, calibpath, atmopath, alpaopath]:
    if not op.exists(p):
        makedirs(p)

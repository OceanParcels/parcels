from parcels import (FieldSet, JITParticle, AdvectionRK4, plotTrajectoriesFile)
from parcels.particleset_vectorized import ParticleSet
from datetime import timedelta as delta
import numpy as np
import pytest
from os import path
from parcels.tools.loggers import logger
import sys


def create_outputfiles(dir):
    datafile = path.join(path.dirname(__file__), 'test_data', 'testfields')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    pset = ParticleSet(fieldset=fieldset, lon=[], lat=[], pclass=JITParticle)
    npart = 10
    delaytime = delta(hours=1)
    endtime = delta(hours=24)
    x = 3. * (1. / 1.852 / 60)
    y = (fieldset.U.lat[0] + x, fieldset.U.lat[-1] - x)
    lat = np.linspace(y[0], y[1], npart)

    fp = dir.join("DelayParticle.nc")
    output_file = pset.ParticleFile(name=fp, outputdt=delaytime)

    for t in range(npart):
        time = 0 if len(pset) == 0 else pset[0].time
        pset.add(ParticleSet(pclass=JITParticle, lon=x, lat=lat[t], fieldset=fieldset, time=time))
        pset.execute(AdvectionRK4, runtime=delaytime, dt=delta(minutes=5),
                     output_file=output_file)

    pset.execute(AdvectionRK4, runtime=endtime-npart*delaytime,
                 dt=delta(minutes=5), output_file=output_file)
    output_file.close()

    return fp


@pytest.mark.parametrize('mode', ['2d', '3d', 'movie2d', 'hist2d'])
def test_plotting(mode, tmpdir):
    if mode == '3d' and sys.platform in ['linux', 'linux2']:
        logger.info('Skipping 3d test in linux Travis, since it fails to find display to connect')
        return
    fp = create_outputfiles(tmpdir)
    plotTrajectoriesFile(fp, mode=mode, show_plt=False)

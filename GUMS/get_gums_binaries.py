from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt
import tqdm, h5py


# Download from Gaia archive
job = Gaia.launch_job_async("select * from gaiaedr3.gaia_universe_model where barycentric_distance < 100")
gums = job.get_results()
gums.sort('source_extended_id')
gums['system_id'] = np.array(gums['source_extended_id'], dtype='S17')
gums.add_index('source_extended_id')

# Get multiplicity of sources
systems = np.array([sei[:17] for sei in gums['source_extended_id'] if '+' not in sei])
multiplicity = {sys_id:multiplicity for sys_id,multiplicity  in zip(*np.unique(systems, return_counts=True))}

# Get binaries
binaries = {'system_id':[]}
star_keys = ['mass','mag_g','mag_bp','mag_rp','mag_rvs','teff','logg','radius','vsini','variability_type','variability_amplitude','variability_period','variability_phase']
system_keys = ['ra','dec','barycentric_distance','pmra','pmdec','radial_velocity','population','age','feh','alphafe']
orbit_keys = ['semimajor_axis','eccentricity','inclination','longitude_ascending_node','orbit_period','periastron_date','periastron_argument']

for key in star_keys:
  binaries['primary_'+key] = []
  binaries['secondary_'+key] = []
for key in system_keys+orbit_keys:
  binaries[key] = []

for i in tqdm.tqdm(range(len(gums))):
  if multiplicity[gums['system_id'][i]] == 2 and gums['source_extended_id'][i]==gums['system_id'][i]+'+     ':

    node = gums[i]
    primary = gums[i+1]
    secondary = gums[i+2]

    assert (primary['source_extended_id'] == gums['system_id'][i]+'A     ') | (primary['source_extended_id'] == gums['system_id'][i]+'AV    ')
    assert (secondary['source_extended_id'] == gums['system_id'][i]+'B     ') | (secondary['source_extended_id'] == gums['system_id'][i]+'BV    ')

    binaries['system_id'].append(node['system_id'])

    for key in star_keys:
      binaries['primary_'+key].append(primary[key])
      binaries['secondary_'+key].append(secondary[key])

    for key in system_keys:
      binaries[key].append(node[key])

    for key in orbit_keys:
      binaries[key].append(secondary[key])


# Get single sources
sys_id, multiplicity = np.unique(systems, return_counts=True)
single_source = np.intersect1d(np.array(gums['source_extended_id'], dtype='S17'),
                               sys_id[multiplicity==1], return_indices=True)[1]

star_keys = ['mass','mag_g','mag_bp','mag_rp','mag_rvs','teff','logg','radius','vsini','variability_type','variability_amplitude','variability_period','variability_phase']
system_keys = ['ra','dec','barycentric_distance','pmra','pmdec','radial_velocity','population','age','feh','alphafe']
orbit_keys = ['semimajor_axis','eccentricity','inclination','longitude_ascending_node','orbit_period','periastron_date','periastron_argument']
singles = {'system_id':np.array(gums['source_extended_id'], dtype='S17')[single_source]}

for key in star_keys:
  singles['primary_'+key] = gums[key][single_source]
  singles['secondary_'+key] = np.zeros(len(single_source))+np.nan

for key in system_keys:
  singles[key] = gums[key][single_source]

for key in orbit_keys:
  singles[key] = np.zeros(len(single_source))+np.nan

# Convert types
dtypes = {'system_id':'S17',
          'primary_variability_type':'S32',
          'secondary_variability_type':'S32'}
data = {}
for key in singles.keys():
    data[key] = np.hstack((singles[key], binaries[key]))
for key in dtypes:
    data[key] = data[key].astype(dtypes[key])
data['binary'] = np.hstack((np.zeros(len(singles['system_id']), dtype=bool),
                            np.ones(len(binaries['system_id']), dtype=bool)))

# Save
with h5py.File('/data/vault/asfe2/Conferences/EDR3_workshop/gums_sample.h', 'w') as hf:
    for key in data.keys():
        print(key)
        hf.create_dataset(key, data=data[key])

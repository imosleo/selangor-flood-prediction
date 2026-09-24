import os
import pandas as pd
import numpy as np
import xarray as xr
import requests

os.chdir('C:\\Users\\Hunts\\OneDrive\\Desktop\\Rainfall Data')

ds = pd.read_csv('subset_GPM_3IMERGDL_06_20230404_073241_.txt', header=None, sep='\r\n', engine='python')[0]

for file in range(2, len(ds)):
    url = ds[file]

    filename = 'data'+str(file-1)+'.nc4'

    result = requests.get(url)
    try:
        result.raise_for_status()
        f = open(filename, 'wb')
        f.write(result.content)
        f.close()
        print('File '+str(file-1)+'/'+str(len(ds)-2)+' downloaded')
    except:
        print('requests.get() returned an error code' + str(result.status_code))

    dowloadedData = xr.open_mfdataset(filename)
    dataset = dowloadedData.to_dataframe()
    dowloadedData.close()
    dataset = dataset.reset_index()
    nv0 = dataset[dataset.nv != 1.0]

    HqPrecip = nv0.drop(columns=['nv','precipitationCal','time_bnds','lon','lat']).reset_index(drop=True)
    PrecipCal = nv0.drop(columns=['nv','HQprecipitation','time_bnds','lon','lat']).reset_index(drop=True)

    HqPrecips = [HqPrecip['HQprecipitation'].tolist()]
    PrecipCals = [PrecipCal['precipitationCal'].tolist()]

    HqRow = pd.DataFrame({
        'time':HqPrecip['time'][0],
        'HqPrecips':HqPrecips
        })
    PrecipRow = pd.DataFrame({
        'time':PrecipCal['time'][0],
        'PrecipCals':PrecipCals
        })

    if file-1 == 1:
        HqRow.to_csv('HQprecipitation Data Test.csv', mode='a', index=False, header=True)
        PrecipRow.to_csv('precipitationCal Data Test.csv', mode='a', index=False, header=True)
    else:
        HqRow.to_csv('HQprecipitation Data Test.csv', mode='a', index=False, header=False)
        PrecipRow.to_csv('precipitationCal Data Test.csv', mode='a', index=False, header=False)

    if os.path.exists(filename):
        os.remove(filename)
        print('Row '+str(file-1)+'/'+str(len(ds)-2)+' appended')
    else:
        print('File deletion failed')

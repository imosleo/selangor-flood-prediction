import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

nowImergL = datetime.now() - timedelta(days=1, hours=22)

# Sample url
# 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.06' + '/2023/' + '02/' + '3B-DAY-L.MS.MRG.3IMERG.' + '20230217' + \
# '-S000000-E235959.V06.nc4.nc4?HQprecipitation[0:0][2806:2820][925:939],time,lon[2806:2820],lat[925:939]'

# url constructor
dirName = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.06'
year = nowImergL.strftime("/%Y/")
month = nowImergL.strftime("%m/")
prodName = '3B-DAY-L.MS.MRG.3IMERG.'
date = nowImergL.strftime("%Y%m%d")
queryEnd = '-S000000-E235959.V06.nc4.nc4?precipitationCal[0:0][2806:2820][925:939],HQprecipitation[0:0][2806:2820][925:939],time,lon[2806:2820],lat[925:939]'

url = dirName + year + month + prodName + date + queryEnd

# write to file
try:
    result = requests.get(url)
except:
    print('Data not yet uploaded')

f = open('IMERG_dump.nc4', 'wb')
f.write(result.content)
f.close()

# read downloaded file
dowloadedData = xr.open_mfdataset('IMERG_dump.nc4', engine='netcdf4')
dataset = dowloadedData.to_dataframe()
dowloadedData.close()
dataset = dataset.reset_index()

HqPrecips = [dataset['HQprecipitation'].tolist()]
HqRow = pd.DataFrame({
    'time':dataset['time'][0],
    'HqPrecips':HqPrecips
    })

PrecipCals = [dataset['precipitationCal'].tolist()]
PrecipRow = pd.DataFrame({
        'time':dataset['time'][0],
        'PrecipCals':PrecipCals
        })

HqRow.to_csv('HQprecipitation Data Test.csv', index=False, header=True)
PrecipRow.to_csv('precipitationCal Data Test.csv', index=False, header=True)
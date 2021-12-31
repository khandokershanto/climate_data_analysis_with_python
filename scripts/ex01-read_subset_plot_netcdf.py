#       Open and process netcdf4 data using xarray

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

#read data
path = r"H:\Python project 2021\climate_data_analysis_with_python\data\HadISST_sst.nc"
nc_file = xr.open_dataset(path)

# plot a single month
nc_file['sst'][-1].plot(cmap = 'jet',vmax = 35,vmin = -10)

# making colimatologies by averaging all the month
nc_clim = nc_file['sst'].mean('time',keep_attrs = True)

nc_clim.plot(cmap = 'jet',vmax = 35,vmin = -10)
nc_clim.attrs['units'] = 'degree C'

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))

nc_clim.plot.contourf(ax = ax,levels = np.arange(-10,36,2),extend = 'max', transform = ccrs.PlateCarree(),cbar_kwargs = {'label': nc_clim.units},cmap = 'jet')

ax.coastlines()
plt.show()

# Subset the nc file with dimension (lat,lon,time)


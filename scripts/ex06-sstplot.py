import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

# ----- Parameter setting ------
# == Figure name ==

# == netcdf file name and location"
fnc = 'data/oisst_monthly.nc'
dmask = xr.open_dataset('data/lsmask.nc')
#print(dmask)

ds = xr.open_dataset(fnc)
#print(ds)

#demo plot
#ds['sst'][-1].plot.contourf()

# mask the land area
#ds_masked = ds.where(dmask.mask.isel(time =0) == 1) #mask land from the total nc file
#ds_masked['sst'][-1].plot() # demo plot

sst = ds.sst.where(dmask.mask.isel(time=0) == 1) # mask land from the sst variable [^.^]
sst = sst.sel(time=slice('1982-01-01','2020-12-01'))
bob = sst.where((5 <= sst.lat) & (sst.lat <= 25) & (80 <= sst.lon) & (sst.lon <= 100),drop=True)
time = sst.time

clm = bob.sel(time=slice('1982-01-01','2020-12-01')).groupby('time.month').mean(dim='time')

clm[1].plot()

#define custom seasons <- winter = NDJF-11,12,1,2 ; spring = MAM-3,4,5 ; Summer = JJA-6,7,8 ; Fall = SO-9,10

winter = bob.time.dt.month.isin([1,2,11,12])
winter_clim = bob.sel(time = winter).mean('time')

spring = bob.time.dt.month.isin([3,4,5])
spring_clim = bob.sel(time = spring).mean('time')

summer = bob.time.dt.month.isin([6,7,8])
summer_clim = bob.sel(time = summer).mean('time')

fall = bob.time.dt.month.isin([9,10])
fall_clim = bob.sel(time = fall).mean('time')

season = xr.concat([winter_clim,spring_clim,summer_clim,fall_clim],dim='season')
seasonplot = season.plot(x = 'lon',y = 'lat',col = 'season',col_wrap=2)


sst[0].plot.contourf()
img_extent = [80,100,5,25]
plt.figure(figsize=[10,8])
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent((80, 100, 5, 25))
ax.plot.(winter_clim, transform=ccrs.PlateCarree())
ax.add_feature(cf.LAND, color='grey')
ax.coastlines(color='black', linewidth=1)
plt.show()

# Use geocat.viz.util convenience function to set axes limits & tick values
gvutil.set_axes_limits_and_ticks(ax,
                                 xlim=(80, 100),
                                 ylim=(5, 25),
                                 xticks=np.arange(80, 101, 5),
                                 yticks=np.arange(5, 26, 5))

# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax, labelsize=15,x_minor_per_major=5,y_minor_per_major=5)

gvutil.add_lat_lon_ticklabels(ax)


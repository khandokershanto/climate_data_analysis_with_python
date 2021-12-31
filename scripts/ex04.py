import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
from datetime import datetime
# geocat for more aesthetic plot
import geocat.viz.util as gvutil
#import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps


# ----- Parameter setting ------
# == Figure name ==

# == netcdf file name and location"
fnc = 'data/oisst_monthly.nc'
dmask = xr.open_dataset('data/lsmask.nc')
print(dmask)

ds = xr.open_dataset(fnc)
print(ds)

#demo plot
ds['sst'][-1].plot()

# mask the land area
#ds_masked = ds.where(dmask.mask.isel(time =0) == 1) #mask land from the total nc file
#ds_masked['sst'][-1].plot() # demo plot

sst = ds.sst.where(dmask.mask.isel(time=0) == 1) # mask land from the sst variable [^.^]
sst = sst.sel(time=slice('1982-01-01','2020-12-01'))

time = sst.time

# anomaly determination
clm = sst.sel(time=slice('1982-01-01','2020-12-01')).groupby('time.month').mean(dim='time')
anm = (sst.groupby('time.month') - clm)

time = anm.time
# -------------------------------------------------


def areaave(indat, latS, latN, lonW, lonE):
    lat = indat.lat
    lon = indat.lon

    if (((lonW < 0) or (lonE < 0)) and (lon.values.min() > -1)):
        anm = indat.assign_coords(lon=((lon + 180) % 360 - 180))
        lon = ((lon + 180) % 360 - 180)
    else:
        anm = indat

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

    #  print(iplat)
    #  print(iplon)
    odat = anm.sel(lat=iplat, lon=iplon).mean(("lon", "lat"), skipna=True)
    return (odat)

# bob sst
ts_bob = areaave(sst,5,25,80,100)

#plot bob time series
fig = plt.figure(figsize=[8,5])
ax1 = fig.add_subplot(111)

ax1.set_title('SST time series in Bay of Bengal')
ax1.plot(time, ts_bob, '-',  linewidth=1)
ax1.set_ylabel('SSTA')
ax1.axis(xmin=pd.Timestamp("1982-01"), xmax=pd.Timestamp("2020-12"))

#ax1.grid(True)
plt.draw()
plt.tight_layout()
plt.savefig("bob_timeseries.png",dpi = 300)

# global vs bob sst anomaly
glob_anom = anm.mean(('lon','lat'),skipna = True)
bob_anom = areaave(anm,5,25,80,100)

xr.corr(glob_anom,bob_anom)

#   plot
fig = plt.figure(figsize=[8,5])
ax1 = fig.add_subplot(111)

ax1.set_title('Globally averaged SSTA & BOB SSTA (OISST v2)')
ax1.plot(time, glob_anom, '-',  linewidth=1)
ax1.plot(time, bob_anom, '-',  linewidth=1)
ax1.tick_params(length = 7,right=True,labelsize=12)
ax1.legend(['Globally averaged','BoB averaged'])
ax1.set_ylabel('SSTA (°C)',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.text(pd.to_datetime('2000-11-01'),-0.6,'Correlation Coefficient = 0.56',fontsize=12)

#ax1.axis(xmin=pd.Timestamp("1982-01"), xmax=pd.Timestamp("2020-12"))
fig.autofmt_xdate()

# Show the plot
plt.draw()
plt.tight_layout()
plt.savefig("bobvsgloobalanom.png",dpi = 300)

#nino 3.4 and dipole mode index plot together
nino = areaave(anm,-5,5,-170,-120)

#IOD west: 50 ° E to 70 ° E and 10 ° S to 10 ° N.
iod_west = areaave(anm,-10,10,50,70)

# IOD east: 90 ° E to 110 ° E and 10 ° S to 0 ° S.
iod_east = areaave(anm,-10,0,90,110)

dmi = iod_west - iod_east

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.set_title('BoB anomaly with repect to ENSO')
ax1.plot(time, bob_anom, '-',  linewidth=1)
ax1.plot(time, nino, '-',  linewidth=1)
ax1.legend(['BoB anomaly','Nino3.4 region'])
ax1.set_ylabel('SSTA [C]')

ax2.set_title('BoB anomaly with respect to IOD')
ax2.plot(time, bob_anom, '-',  linewidth=1)
ax2.plot(time, dmi, '-',  linewidth=1)
ax2.legend(['BoB anomaly','Dipole Mode Index'])

ax2.set_ylabel('SSTA [C]')

#ax1.axis(xmin=pd.Timestamp("1982-01"), xmax=pd.Timestamp("2020-12"))
ax1.grid(True)
ax2.grid(True)
fig.autofmt_xdate()

# Show the plot
plt.draw()
plt.tight_layout()
plt.savefig("nino-bob-dmi.png",dpi = 300)



##        ~~~~~                      Plotting  nino index                     ~~~~~~~~~~~~~~~~~~~~

itime = np.arange(time.size)
# calculating nino 3.4 index
nino = areaave(anm,-5,5,-170,-120)
rnino=nino.rolling(time=7, center=True).mean('time')
#nino standard
ninoSD=nino/nino.std(dim='time')
rninoSD=ninoSD.rolling(time=7, center=True).mean('time')


#    -- --  --  --  --  --  --  --  -   --  -       --  --- --  -   --      -   --  -   -   --  -   -
# --                                      figure plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
# Create a list of colors based on the color bar values
colors = ['C1' if (value > 0) else 'C0' for value in ninoSD]

# Plot bar chart
ax1.bar(itime, nino, align='edge', edgecolor="none", color=colors, width=1.0)
ax1.plot(itime, rnino, color="black", linewidth=1.5)
ax1.legend(['7-month running mean'])

ax2.bar(itime, ninoSD, align='edge', edgecolor="none", color=colors, width=1.0)
ax2.plot(itime, rninoSD, color="black", linewidth=1.5)
#################################################################################
# Use geocat.viz.util convenience function to set axes parameters
ystr = 1985
yend = 2020
dyr = 5
ist, = np.where(time == pd.Timestamp(year=ystr, month=1, day=1) )
iet, = np.where(time == pd.Timestamp(year=yend, month=1, day=1) )
gvutil.set_axes_limits_and_ticks(ax1,
                                 ylim=(-3, 3.5),
                                 yticks=np.linspace(-3, 3, 7),
                                 yticklabels=np.linspace(-3, 3, 7),
                                 xlim=(itime[0], itime[-1]),
                                 xticks=itime[ist[0]:iet[0]+1:12*dyr],
                                 xticklabels=np.arange(ystr, yend+1, dyr) )

gvutil.set_axes_limits_and_ticks(ax2,
                                 ylim=(-3, 3.5),
                                 yticks=np.linspace(-3, 3, 7),
                                 yticklabels=np.linspace(-3, 3, 7),
                                 xlim=(itime[0], itime[-1]),
                                 xticks=itime[ist[0]:iet[0]+1:12*dyr],
                                 xticklabels=np.arange(ystr, yend+1, dyr) )

###
# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax1,
                             x_minor_per_major=5,
                             y_minor_per_major=5,
                             labelsize=12)

gvutil.add_major_minor_ticks(ax2,
                             x_minor_per_major=5,
                             y_minor_per_major=5,
                             labelsize=12)
# Use geocat.viz.util convenience function to set titles and labels
gvutil.set_titles_and_labels(ax1,
                             maintitle="SSTA in Nino3.4 region",
                             ylabel='Anomalies',
                             maintitlefontsize=18,
                             labelfontsize=15)

gvutil.set_titles_and_labels(ax2,
                             maintitle="Nino3.4 Index",
                             ylabel='Standardized',
                             xlabel='Year',
                             maintitlefontsize=18,
                             labelfontsize=15)


# Show the plot
#ax1.grid(True)
#ax2.grid(True)

plt.draw()
plt.tight_layout()
plt.savefig("nino3.4.png",dpi=300)









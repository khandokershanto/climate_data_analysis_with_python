import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geocat.viz.util as gvutil

path = r'H:\Python project 2021\climate_data_analysis_with_python\data\sst.mnmean.nc'
ds= xr.open_dataset(path)
# time slicing
sst = ds.sst.sel(time=slice('1920-01-01','2020-12-01'))
# anomaly with respect to 1971-2000 period
clm = ds.sst.sel(time=slice('1971-01-01','2000-12-01')).groupby('time.month').mean(dim='time')
anm = (sst.groupby('time.month') - clm)

time = anm.time
itime=np.arange(time.size)
def wgt_areaave(indat, latS, latN, lonW, lonE):
  lat=indat.lat
  lon=indat.lon

  if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
     anm=indat.assign_coords(lon=( (lon + 180) % 360 - 180) )
     lon=( (lon + 180) % 360 - 180)
  else:
     anm=indat

  iplat = lat.where( (lat >= latS ) & (lat <= latN), drop=True)
  iplon = lon.where( (lon >= lonW ) & (lon <= lonE), drop=True)

#  print(iplat)
#  print(iplon)
  wgt = np.cos(np.deg2rad(lat))
  odat=anm.sel(lat=iplat,lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
  return(odat)

# bob sst
bob_anm = wgt_areaave(anm,5,25,80,100)
bob_ranm = bob_anm.rolling(time=7, center=True).mean('time')
##
# Create a list of colors based on the color bar values
colors = ['C1' if (value > 0) else 'C0' for value in bob_anm]

fig = plt.figure(figsize=[8,5])
ax1 = fig.add_subplot(111)

# Plot bar chart
ax1.bar(itime, bob_anm, align='edge', edgecolor="none", color=colors, width=1.0)
ax1.plot(itime, bob_ranm, color="black", linewidth=1.5)
ax1.legend(['7-month running mean'],fontsize=12)

# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax1,
                             x_minor_per_major=4,
                             y_minor_per_major=5,
                             labelsize=12)
# Use geocat.viz.util convenience function to set axes parameters
ystr = 1920
yend = 2020
dyr = 20
ist, = np.where(time == pd.Timestamp(year=ystr, month=1, day=1) )
iet, = np.where(time == pd.Timestamp(year=yend, month=1, day=1) )
gvutil.set_axes_limits_and_ticks(ax1,
                                 ylim=(-1.5, 1),
                                 yticks=np.linspace(-1.5, 1, 6),
                                 yticklabels=np.linspace(-1.5, 1, 6),
                                 xlim=(itime[0], itime[-1]),
                                 xticks=itime[ist[0]:iet[0]+1:12*dyr],
                                 xticklabels=np.arange(ystr, yend+1, dyr))

# Use geocat.viz.util convenience function to set titles and labels
gvutil.set_titles_and_labels(ax1,
                             maintitle="SSTA in BoB (ERSST)",
                             ylabel='Anomalies',
                             xlabel= 'Year',
                             maintitlefontsize=18,
                             labelfontsize=15)
plt.tight_layout()
plt.savefig("bob_anomalies.png",dpi = 300)

########## BoB SST with respect to ENSO and IOD (ERSST)
#nino 3.4 and dipole mode index plot together
nino = wgt_areaave(anm,-5,5,-170,-120)
nino = nino.rolling(time=7, center=True).mean('time')

#IOD west: 50 ° E to 70 ° E and 10 ° S to 10 ° N.
iod_west = wgt_areaave(anm,-10,10,50,70)

# IOD east: 90 ° E to 110 ° E and 10 ° S to 0 ° S.
iod_east = wgt_areaave(anm,-10,0,90,110)

dmi = iod_west - iod_east
dmi = dmi.rolling(time=7, center=True).mean('time')

###             Figure Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.set_title('BoB anomaly with repect to ENSO')
ax1.plot(time, bob_ranm, '-',  linewidth=1)
ax1.plot(time, nino, '-',  linewidth=1)
ax1.tick_params(length = 7,right=True,labelsize=12)
ax1.legend(['BoB anomaly','Nino3.4 Index'],fontsize=12,frameon=False)
ax1.set_ylabel('SSTA (°C)',fontsize=12)

ax2.set_title('BoB anomaly with respect to IOD')
ax2.plot(time, bob_ranm, '-',  linewidth=1)
ax2.plot(time, dmi, '-',  linewidth=1)
ax2.tick_params(length = 7,right=True,labelsize=12)
ax2.legend(['BoB anomaly','Dipole Mode Index'],fontsize=12,frameon=False)
ax2.set_ylabel('SSTA (°C)',fontsize=12)



# Show the plot
plt.draw()
plt.tight_layout()
plt.savefig("nino-bob-dmi.png",dpi = 300)

#######################                  (Ploting Nino 3.4 Index)

nino = wgt_areaave(anm,-5,5,-170,-120)
rnino = nino.rolling(time=7, center=True).mean('time')
#nino standard
ninoSD=nino/nino.std(dim='time')
rninoSD=ninoSD.rolling(time=7, center=True).mean('time')

#    -- --  --  --  --  --  --  --  -   --  -       --  --- --  -   --      -   --  -   -   --  -   -
# --                                      figure plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
# Create a list of colors based on the color bar values
colors = ['C1' if (value > 0) else 'C0' for value in ninoSD]

# Plot bar chart
ax1.bar(itime, nino, align='edge', edgecolor="none", color=colors, width=1.0)
ax1.plot(itime, rnino, color="black", linewidth=1.5)
ax1.legend(['7-month running mean'],fontsize=12,frameon=False)

ax2.bar(itime, ninoSD, align='edge', edgecolor="none", color=colors, width=1.0)
ax2.plot(itime, rninoSD, color="black", linewidth=1.5)

# Use geocat.viz.util convenience function to set axes parameters
ystr = 1920
yend = 2020
dyr = 20
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

# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax1,
                             x_minor_per_major=4,
                             y_minor_per_major=5,
                             labelsize=12)

gvutil.add_major_minor_ticks(ax2,
                             x_minor_per_major=4,
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

plt.draw()
plt.tight_layout()
plt.savefig("nino3.4_ERSST.png",dpi=300)

###############                     (Ploting DMI Index)
iod_west = wgt_areaave(anm,-10,10,50,70)
# IOD east: 90 ° E to 110 ° E and 10 ° S to 0 ° S.
iod_east = wgt_areaave(anm,-10,0,90,110)
dmi = iod_west - iod_east
rdmi = dmi.rolling(time=7, center=True).mean('time')

colors = ['C1' if (value > 0) else 'C0' for value in dmi]

fig = plt.figure(figsize=[8,5])
ax1 = fig.add_subplot(111)

# Plot bar chart
ax1.bar(itime, dmi, align='edge', edgecolor="none", color=colors, width=1.0)
ax1.plot(itime, rdmi, color="black", linewidth=1.5)
ax1.legend(['7-month running mean'],fontsize=12,frameon=False)

# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax1,
                             x_minor_per_major=4,
                             y_minor_per_major=5,
                             labelsize=12)
# Use geocat.viz.util convenience function to set axes parameters
ystr = 1920
yend = 2020
dyr = 20
ist, = np.where(time == pd.Timestamp(year=ystr, month=1, day=1) )
iet, = np.where(time == pd.Timestamp(year=yend, month=1, day=1) )
gvutil.set_axes_limits_and_ticks(ax1,
                                 ylim=(-1.5, 1.90),
                                 yticks=np.linspace(-1, 1.5, 6),
                                 yticklabels=np.linspace(-1, 1.5, 6),
                                 xlim=(itime[0], itime[-1]),
                                 xticks=itime[ist[0]:iet[0]+1:12*dyr],
                                 xticklabels=np.arange(ystr, yend+1, dyr))

# Use geocat.viz.util convenience function to set titles and labels
gvutil.set_titles_and_labels(ax1,
                             maintitle=" Dipole Mode Index",
                             ylabel='Anomalies',
                             xlabel= 'Year',
                             maintitlefontsize=18,
                             labelfontsize=15)
plt.tight_layout()
plt.savefig("dmi_ersst.png",dpi = 300)

###                      (Global vs BoB time Series -ERSST v5)
# global vs bob sst anomaly
glob_anom = anm.mean(('lon','lat'),skipna = True)
glob_anom_ra = glob_anom.rolling(time=12, center=True).mean('time')
bob_anm = wgt_areaave(anm,5,25,80,100)
bob_anm_ra = bob_anm.rolling(time=12, center=True).mean('time')

xr.corr(glob_anom_ra,bob_anm_ra)

#   plot
fig = plt.figure(figsize=[8,5])
ax1 = fig.add_subplot(111)

ax1.set_title('Global SSTA & BOB SSTA with 1 year moving average (ERSST v5)')
ax1.plot(time, glob_anom_ra, '-',  linewidth=1)
ax1.plot(time, bob_anm_ra, '-',  linewidth=1)
ax1.tick_params(length = 7,right=True,labelsize=12)
ax1.legend(['Globally averaged','BoB averaged'],fontsize=12,frameon=False)
ax1.set_ylabel('SSTA (°C)',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.text(pd.to_datetime('1975-01-01'),-0.8,'Correlation Coefficient = 0.89',fontsize=12)

#ax1.axis(xmin=pd.Timestamp("1982-01"), xmax=pd.Timestamp("2020-12"))


# Show the plot
plt.draw()
plt.tight_layout()
plt.savefig("bobvsgloobalanom_ersst.png",dpi = 300)


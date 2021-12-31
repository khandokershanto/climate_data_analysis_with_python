import  numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=[11,8])
ax1 = fig.add_subplot(111)

ax1.axis([0,360,-1,1])
ax1.tick_params(direction = 'out',which='both')
ax1.set_xlabel('Degrees'),ax1.set_ylabel('Amplitude')
ax1.set_xticks(np.arange(0,361,30)),ax1.set_yticks(np.arange(-1,1.1,0.25))

xpts = np.arange(0,361,1)
ax1.plot(xpts,np.sin(np.radians(xpts)),label='Sine',color = 'blue')
ax1.plot(xpts,np.cos(np.radians(xpts)),label = 'Cosine',color = 'red')
ax1.legend(loc='lower left')

##############################################
#     line plot
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MultipleLocator

# load nc dataset
nc = xr.open_dataset('data/HadISST_sst.nc')

# subset bob region (lat 5-25 & lon 80-100 & time)
bob = nc.where((5 <= nc.latitude) & (nc.latitude <= 25) & (80 <= nc.longitude) & (nc.longitude <= 100),drop=True)
bob = bob.sel(time = slice('1920-01-16','2019-12-16'))

bob['sst'][0].plot(cmap = 'jet')

###
time = bob['time'][:]
sst = bob['sst'][:]






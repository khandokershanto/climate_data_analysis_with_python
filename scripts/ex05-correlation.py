N = 100
ds = xr.Dataset(
    data_vars={
        'x': ('t', np.random.randn(N)),
        'y': ('t', np.random.randn(N))
    },
    coords={
        't': range(N)
    }
)
xr.corr(ds['y'], ds['x'])

#

ds_dask = ds.chunk({"t": 10})

yy = xr.corr(ds['y'], ds['y']).to_numpy()
yy_dask = xr.corr(ds_dask['y'], ds_dask['y']).to_numpy()
yx = xr.corr(ds['y'], ds['x']).to_numpy()
yx_dask = xr.corr(ds_dask['y'], ds_dask['x']).to_numpy()
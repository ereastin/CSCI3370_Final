import xarray as xr
import earthaccess
import boto3

def main():
    boto3.setup_default_session(region_name='us-west-2')
    s3_client = boto3.client('s3')
    if (s3_client.meta.region_name != 'us-west-2'):
        print('failed connecting to correct region')

    # Authenticate using Earthdata Login prerequisite files
    auth = earthaccess.login()

    # Search for the granule by DOI
    results = earthaccess.search_data(
        short_name='M2I3NVASM',
        temporal=("2019-03-13", "2019-03-13"),
        bounding_box=(-110, 24, -70, 52)
    )

    fn = earthaccess.open(results)

    ds = xr.open_mfdataset(
        fn,
        drop_variables=['CLOUD', 'EPV', 'DELP', 'O3', 'RH', 'SLP', 'PHIS', 'PL', 'PS', 'QI', 'QL'],
        chunks='auto',
        engine='h5netcdf'
    )
    # self.merra_lats = slice(14, 53.5)  -> (25, 51)
    # self.merra_lons = slice(-140, -50.625)  -> (-110, -70)
    ds = ds.sel(lon=slice(-135, -45.625), lat=slice(18.5, 58), lev=slice(1, 35))

if __name__ == '__main__':
    main()

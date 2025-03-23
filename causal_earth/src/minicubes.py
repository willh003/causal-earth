import xarray as xr
import numpy as np
import torch
from typing import Protocol

def extract_all_rgb(minicube : "Minicube") -> torch.Tensor:
    """The available RGB images in the minicube.
    `torch.Tensor: float32 (time, rgb, lat, long)`"""
    avail_mask = minicube.s2_avail.notnull()
    red = minicube.s2_B04[avail_mask].values
    green = minicube.s2_B03[avail_mask].values
    blue = minicube.s2_B02[avail_mask].values
    rgb = torch.from_numpy(np.stack([red, green, blue], axis=1))
    return rgb

class Minicube(Protocol):
    """
    A protocol representing a Minicube from the EarthNet2021x dataset.
    https://www.earthnet.tech/en21/ds-specifications/ 
    """

    ### NOTE: Below are all the original fields from the dataset.
    def time(self) -> xr.DataArray:
        """`xr.DataArray: datetime64[ns] (time,)`"""
        ...
    def lon(self) -> xr.DataArray:
        """`xr.DataArray: float64 (lon,)`"""
        ...
    def lat(self) -> xr.DataArray:
        """`xr.DataArray: float64 (lat,)`"""
        ...
    def s2_avail(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`"""
        ...
    def s2_B02(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: linear
            description: Blue
        """
        ...
    def s2_B03(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: linear
            description: Green
        """
        ...
    def s2_B04(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: linear
            description: Red
        """
        ...
    def s2_B8A(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: linear
            description: Near infrared (NIR) Narrow
        """
        ...
    def alos_dem(self) -> xr.DataArray:
        """`xr.DataArray: float32 (lat, lon,)`
        
        Attributes:
            provider: ALOS World 3D-30m
            interpolation_type: linear
            description: Elevation data.
            units: metre
        """
        ...
    def cop_dem(self) -> xr.DataArray:
        """`xr.DataArray: float32 (lat, lon,)`
        
        Attributes:
            provider: Copernicus DEM GLO-30
            interpolation_type: linear
            description: Elevation data.
            units: metre
        """
        ...
    def esawc_lc(self) -> xr.DataArray:
        """`xr.DataArray: float32 (lat, lon,)`
        
        Attributes:
            provider: ESA Worldcover
            interpolation_type: nearest
            description: Land cover classification
            classes:
                10 - Tree cover
                20 - Shrubland
                30 - Grassland
                40 - Cropland
                50 - Built-up
                60 - Bare / sparse vegetation
                70 - Snow and Ice
                80 - Permanent water bodies
                90 - Herbaceous wetland
                95 - Mangroves
                100 - Moss and lichen
        """
        ...
    def geom_cls(self) -> xr.DataArray:
        """`xr.DataArray: float32 (lat, lon,)`
        
        Attributes:
            provider: Geomorpho90m
            interpolation_type: nearest
            description: Geomorphon classes. Original resolution ~90m. For more see: https://www.nature.com/articles/s41597-020-0479-6
            classes:
                1: flat
                2: summit
                3: ridge
                4: shoulder
                5: spur
                6: slope
                7: hollow
                8: footslope
                9: valley
                10: depression
        """
        ...
    def s2_SCL(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: nearest
            description: Scene classification layer
            classes:
                0 - No data
                1 - Saturated / Defective
                2 - Dark Area Pixels
                3 - Cloud Shadows
                4 - Vegetation
                5 - Bare Soils
                6 - Water
                7 - Clouds low probability / Unclassified
                8 - Clouds medium probability
                9 - Clouds high probability
                10 - Cirrus
                11 - Snow / Ice
        """
        ...
    def s2_mask(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: nearest
            description: sen2flux Cloud Mask
            classes:
                0 - free sky
                1 - cloud
                2 - cloud shadows
                3 - snow
                4 - masked by SCL
        """
        ...
    def latitude_eobs(self) -> xr.DataArray:
        """`xr.DataArray: float32 (,)`
        
        Attributes:
            standard_name: latitude
            long_name: Latitude values
            units: degrees_north
            axis: Y
        """
        ...
    def eobs_tg(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            standard_name: air_temperature
            long_name: mean temperature
            units: Celsius
            provider: E-OBS v23.1
        """
        ...
    def eobs_fg(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            standard_name: wind_speed
            long_name: Ensemble mean wind speed
            units: m/s
            provider: E-OBS v23.1
        """
        ...
    def eobs_hu(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            cell_methods: ensemble: mean
            long_name: mean relative humidity
            standard_name: relative_humidity
            units: %
            provider: E-OBS v23.1
        """
        ...
    def eobs_pp(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            units: hPa
            long_name: sea level pressure
            standard_name: air_pressure_at_sea_level
            provider: E-OBS v23.1
        """
        ...
    def eobs_qq(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            standard_name: surface_downwelling_shortwave_flux_in_air
            long_name: surface downwelling shortwave flux in air
            units: W/m2
            provider: E-OBS v23.1
        """
        ...
    def eobs_rr(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            units: mm
            long_name: rainfall
            standard_name: thickness_of_rainfall_amount
            provider: E-OBS v23.1
        """
        ...
    def eobs_tn(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            standard_name: air_temperature
            long_name: minimum temperature
            units: Celsius
            provider: E-OBS v23.1
        """
        ...
    def eobs_tx(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time,)`
        
        Attributes:
            standard_name: air_temperature
            long_name: maximum temperature
            units: Celsius
            provider: E-OBS v23.1
        """
        ...
    def nasa_dem(self) -> xr.DataArray:
        """`xr.DataArray: float32 (lat, lon,)`
        
        Attributes:
            provider: NASADEM HGT v001
            interpolation_type: linear
            description: Elevation data.
            units: metre
        """
        ...
    def s2_dlmask(self) -> xr.DataArray:
        """`xr.DataArray: float32 (time, lat, lon,)`
        
        Attributes:
            provider: Sentinel 2
            interpolation_type: nearest
            description: Deep Learning Cloud Mask, trained by Vitus Benson on cloudSEN12, leveraging code from César Aybar.
            classes:
                0 - Clear sky
                1 - Thick cloud
                2 - Thin cloud
                3 - Cloud shadow
        """
        ...

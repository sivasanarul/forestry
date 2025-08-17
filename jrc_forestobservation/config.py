from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np

class Config(BaseModel):
    support_folder: Path = Field(..., description="Path to the support directory containing study extents")
    work_dir: Path = Field(..., description="Path to the working directory")
    gpkg_path: Path = Field(..., description="Path to the GPKG tile extents file")
    jrc_forest_site: str
    
    #
    #
    # unique_data = np.unique(gfw_folder_tile_rasterdata)
    #
    #
    #
    # date_dict = decode_alert(unique_data)
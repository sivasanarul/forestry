import datetime
import json
import logging
import os
from time import gmtime
from pathlib import Path
import shutil

import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo.gdalconst import GDT_Byte, GDT_UInt16
from tondor.util.tool import save_raster_template

from config import Config
from tools.utils import df_to_gdf, identify_master_extent, download_file
from src.tondor.util.tool import read_json_to_list, raster2array, save_raster_template, reproject_multibandraster_toextent, read_raster_info
from backscatter_changedetection import apply_datacube

from gfw_detections import download_gfw_data, decode_alert
from cband_forestdegradation import calculate_monthly_3month_composite

WORK_ROOT = Path("/var/tmp")
INTERPOLATE = True

start_year = 2020
end_year = 2025
DETECTION_START_YEAR = 2021

s3_url_base = f"https://s3.waw3-1.cloudferro.com/swift/v1"

REMOTE_BASEOUTPUTDIR = "output"

PROD = False

log = logging.getLogger(__name__)
HLINE = "------------------------------------------"

def init_logging():
    logging.Formatter.converter = gmtime
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt="{levelname} {message}", style="{"))
    root_log = logging.getLogger()
    root_log.addHandler(console_handler)
    root_log.setLevel(logging.NOTSET)
    log.info("Logging has been started.")
    logging.getLogger("fiona").setLevel(logging.INFO)




def main():
    init_logging()
    log.info("Starting JRC Forest Observation analysis.")

    if PROD:
        # Take parameters.
        task_id = int(os.environ["TONDOR_TASK_ID"]) # 444
        kwargs = json.loads(os.environ["TONDOR_KWARGS"])
        #kwargs = json.loads('{"year":"2021", "bioregion":"CZECHIA","epsg":"32633","pixel_size":"10","opt":"yes","sar":"yes","dem_bands":"3", "class_attribute":"V0_L1"}')
        input_archive_roots = os.environ["INPUT_ARCHIVE_ROOTS"].split(":") #['/home/siv0021/gisat/gisat-internal/output']        #
        input_archive_roots = [Path(path) for path in input_archive_roots if len(path.strip()) > 0]
        output_archive_root = Path(os.environ["OUTPUT_ARCHIVE_ROOT"]) #Path('/home/siv0021/gisat/gisat-internal/output')    #
        output_archive_tmp = Path(os.environ["OUTPUT_ARCHIVE_TMP"]) #Path('/home/siv0021/gisat/gisat-internal/output.tmp') #
        training_parent_path = Path("/optcomposite_selection")
        support_path = Path("/support_data")
        tmp_dir = Path("/var/tmp")
    else:
        task_id = 444
        # Input parameters to update when running in pycharm
        kwargs = json.loads('{"sitecode":"TESTEUDRBRAZIL"}')
        input_archive_roots = [Path('/mnt/hddarchive.nfs/output')]
        output_archive_root = Path('/mnt/hddarchive.nfs/output')
        output_archive_tmp = Path('/mnt/hddarchive.nfs/output.tmp')
        tmp_dir = Path("/mnt/hddarchive.nfs/tmp/optcompinterpolate")
        os.makedirs(tmp_dir, exist_ok=True)
        training_parent_path = Path("/mnt/ssdarchive.nfs/support_data/optcomposite_selection")
        support_path = Path("/mnt/ssdarchive.nfs/support_data")

    #########################################################
    sitecode = kwargs["sitecode"].strip()
    log.info(f"Doing analysis for {sitecode}")

    ## Get extent
    config_path = "config.json"
    config_data = read_json_to_list(config_path)
    config = Config(**config_data)
    log.info(f"{config.__dict__}")
    #########################################################
    # find master extents
    master_gdf = gpd.read_file(config.gpkg_path)
    log.info(f"{master_gdf.head()}")
    #########################################################
    # get extents of aoi
    extents_csv = config.support_folder.joinpath("study_extents", "study_extents.csv")
    df = pd.read_csv(extents_csv)
    df_tile = df[df["tile_name"] == sitecode]

    #########################################################
    gdf = df_to_gdf(df_tile)
    overlapping_tiles = identify_master_extent(gdf, master_gdf)
    log.info(f"{overlapping_tiles.tile_id}")
    # https://ies-ows.jrc.ec.europa.eu/iforce/gfc2020/download.py?version=v2&type=tile&lat=S10&lon=W70
    tile_id = overlapping_tiles.tile_id.iloc[0]
    gfw_tile = tile_id
    lat_value = tile_id.split("_")[0]
    lon_value = tile_id.split("_")[1]

    lat_value_hem = lat_value[-1]
    lon_value_hem = lon_value[-1]
    lat_value_extent = int(lat_value[0:-1])
    lon_value_extent = int(lon_value[0:-1])
    log.info(f"{lat_value_hem}{lat_value_extent} - {lon_value_hem}{lon_value_extent}")
    jrc_forest_link = config.jrc_forest_site.format(f"{lat_value_hem}{lat_value_extent}", f"{lon_value_hem}{lon_value_extent}")
    log.info(f"{jrc_forest_link}")
    #########################################################
    jrc_tif = config.work_dir.joinpath(f"jrc_forestlayer_{lat_value_hem}{lat_value_extent}-{lon_value_hem}{lon_value_extent}.tif")
    if not jrc_tif.exists():
        download_file(jrc_forest_link, jrc_tif)
    #########################################################
    mosaic_tifs = []
    for year_item in range(start_year, end_year + 1):
        log.info(f"collecting mosaic for year {year_item}")
        year_mosaic_folder = output_archive_root.joinpath(str(year_item), sitecode, "MOSAIC", "BAC", "ORBS")
        year_mosaic_files = os.listdir(year_mosaic_folder)
        year_mosaic_filepaths = [year_mosaic_folder.joinpath(year_mosaic_fileitem) for year_mosaic_fileitem in year_mosaic_files]
        mosaic_tifs.extend(year_mosaic_filepaths)
    log.info(f"all mosaic files {len(mosaic_tifs)}")

    # find orbit numbers
    mosaic_orb_dict = {}
    orb_list = []
    for mosaic_tif_item in mosaic_tifs:
        orb_number = mosaic_tif_item.name.split("_")[6]
        orb_list.append(orb_number)
    orb_list = list(set(orb_list))
    log.info(f"orb list {orb_list}")

    #
    template_mosaic_raster = None
    mosaic_orb_date_dict = {}
    for orb_item in orb_list:
        mosaic_orb_date_dict[orb_item] = {}
        for mosaic_tif_item in mosaic_tifs:
            mosaic_date = mosaic_tif_item.name.split("_")[1]

            if not mosaic_date in mosaic_orb_date_dict[orb_item].keys():
                mosaic_orb_date_dict[orb_item][mosaic_date] = {"VV": [], "VH": []}
            mosaic_item_pol = mosaic_tif_item.name.split("_")[4]
            if mosaic_item_pol == "VV": mosaic_orb_date_dict[orb_item][mosaic_date]["VV"] = mosaic_tif_item
            if mosaic_item_pol == "VH": mosaic_orb_date_dict[orb_item][mosaic_date]["VH"] = mosaic_tif_item
            template_mosaic_raster = mosaic_tif_item
        log.info(f"Grouped according to orb {orb_item}")

    for orb_item in mosaic_orb_date_dict:
        mosaic_orb_date_dict[orb_item] = dict(
            sorted(
                mosaic_orb_date_dict[orb_item].items(),
                key=lambda x: x[0]  # x[0] is the date key
            )
        )
    log.info("Sorted according to date")
    #########################################################
    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(template_mosaic_raster)


    change_detection_folder = config.work_dir.joinpath("change_detection")
    os.makedirs(change_detection_folder, exist_ok=True)
    tile_forest_mask = change_detection_folder.joinpath(f"{sitecode}_jrc_forestmask.tif")
    if not tile_forest_mask.exists():
        reproject_multibandraster_toextent(jrc_tif, tile_forest_mask,
                                           epsg, pixel_width,
                                           xmin, imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax)

    #########################################################
    #########################################################
    #########################################################
    #### S1 bacskcatter change detection
    for orbit_number, mosaic_date in mosaic_orb_date_dict.items():
        log.info(f"Doing change detection for {orbit_number}")
        year_threshold = f"{DETECTION_START_YEAR}0101"
        mosaic_keys_to_analyse = sorted([k for k in mosaic_date.keys() if int(k) > int(year_threshold)])
        mosaic_keys_to_analyse = mosaic_keys_to_analyse[:-5]
        all_mosaic_keys = list(mosaic_date.keys())

        mcd_detection_dict = {}
        for analysis_index, mosaic_keys_item in enumerate(all_mosaic_keys):
            if not mosaic_keys_item in mosaic_keys_to_analyse: continue

            log.info("------------------------------------------------------------")
            template_raster = None
            log.info(f"analysing {mosaic_keys_item}")
            DEC_changedetection_path = change_detection_folder.joinpath(f"DEC_{sitecode}_{mosaic_keys_item}_CHANGE.tif")
            DEC_changethreshold_path = change_detection_folder.joinpath(f"DEC_{sitecode}_{mosaic_keys_item}_CHANGETHRESHOLD.tif")

            mcd_detection_dict[mosaic_keys_item] = {"mcd": DEC_changedetection_path, "threshold": DEC_changethreshold_path }
            if DEC_changedetection_path.exists() and DEC_changethreshold_path.exists(): continue

            past_start = max(0, analysis_index - 6)
            past_end = analysis_index -1
            future_start = analysis_index
            future_end = analysis_index + 5

            past_keys = all_mosaic_keys[past_start:past_end]
            future_keys = all_mosaic_keys[future_start:future_end]
            log.info(f"{past_keys} <--> {future_keys}")

            filestack_pol_dict = {"VV": [], "VH": []}
            for past_key_item in past_keys + future_keys:
                for mosaic_dateitem, mosaic_pol_paths in mosaic_date.items():
                    if not mosaic_dateitem == past_key_item: continue
                    for pol_item in ["VV", "VH"]:
                        filestack_pol_dict[pol_item].append(mosaic_pol_paths[pol_item])
            log.info(f"{filestack_pol_dict}")

            arraystack_pol_dict = {"VV": [], "VH": []}
            for pol_index, raster_list in filestack_pol_dict.items():
                raster_array_list = []
                for raster_list_item in raster_list:
                    template_raster = raster_list_item
                    raster_array = raster2array(raster_list_item)
                    raster_array_list.append(raster_array)
                arraystack_pol_dict[pol_index] = np.stack(raster_array_list, axis=0)

            DEC_changemask, DEC_threshold = apply_datacube(arraystack_pol_dict)


            save_raster_template(template_raster,  DEC_changedetection_path, DEC_changemask, GDT_Byte, 0)
            save_raster_template(template_raster,  DEC_changethreshold_path, DEC_threshold, GDT_Byte, 0)
    #########################################################
    ### create s1 change detection master
    mcd_zscoreerror = change_detection_folder.joinpath(f"{sitecode}_zscore")
    os.makedirs(mcd_zscoreerror, exist_ok=True)

    mcd_zscoreerror_done_txt = mcd_zscoreerror.joinpath("zscore_done.txt")
    if not mcd_zscoreerror_done_txt.exists():
        FILES_VALUES = []
        for datetime_index, mcd_ai_dict_path in mcd_detection_dict.items():
            mcd_ai_array = raster2array(mcd_ai_dict_path["threshold"])
            mcd_ai_array[np.isnan(mcd_ai_array)] = 0
            FILE_SUM = np.sum(mcd_ai_array)
            FILES_VALUES.append(FILE_SUM)

        zscore_mcd_dict = {}
        FILES_MEAN = np.mean(FILES_VALUES)
        FILES_STD = np.std(FILES_VALUES)
        for datetime_index, mcd_ai_dict_path in mcd_detection_dict.items():
            mcd_path = mcd_ai_dict_path["threshold"]
            mcd_ai_array = raster2array(mcd_path)
            mcd_ai_array[np.isnan(mcd_ai_array)] = 0
            FILE_SUM = np.sum(mcd_ai_array)
            FILE_Z = (FILE_SUM - FILES_MEAN) / FILES_STD
            print(f"zscore = {FILE_Z} - {mcd_path.name}")
            if FILE_Z > 3:
                error_filepath = mcd_zscoreerror.joinpath(mcd_path.name.replace(".tif", "_error.tif"))
                shutil.copy(mcd_path, error_filepath)
            else:
                zscore_mcd_path = mcd_zscoreerror.joinpath(mcd_path.name)
                shutil.copy(mcd_path, zscore_mcd_path)
                zscore_mcd_dict[datetime_index] = zscore_mcd_path
        mcd_zscoreerror_done_txt.write_text("Zscore done")
    else:
        zscore_mcd_dict = {}
        zscore_checked_files = os.listdir(mcd_zscoreerror)
        zscore_checked_tifs = [zscore_checked_file_item for zscore_checked_file_item in zscore_checked_files if zscore_checked_file_item.endswith("CHANGETHRESHOLD.tif")]
        for zscore_checked_tif_item in zscore_checked_tifs:
            datetime_index = zscore_checked_tif_item.split("_")[2]
            zscore_mcd_dict[datetime_index] = mcd_zscoreerror.joinpath(zscore_checked_tif_item)

    #########################################################
    ### create change sum
    master_detection_nomask_path = change_detection_folder.joinpath(f"DEC_{sitecode}_S1_CHANGE_NOMASK.tif")
    master_detection_masked_path = change_detection_folder.joinpath(f"DEC_{sitecode}_S1_CHANGE_MASKED.tif")

    if not master_detection_nomask_path.exists() or not master_detection_masked_path.exists():
        tile_forest_elevation_mask = raster2array(tile_forest_mask)
        tile_forest_elevation_mask[np.isnan(tile_forest_elevation_mask)] = 0

        master_detection_nomask_array = np.zeros_like(tile_forest_elevation_mask)

        for datetime_index, zscore_mcd_path in zscore_mcd_dict.items():
            mcd_ai_array = raster2array(zscore_mcd_path)
            mcd_ai_array[np.isnan(mcd_ai_array)] = 0
            master_detection_nomask_array += mcd_ai_array


        save_raster_template(tile_forest_mask, master_detection_nomask_path, master_detection_nomask_array, GDT_UInt16, 0)

        master_detection_nomask_array[tile_forest_elevation_mask == 0] = 0
        master_detection_nomask_array[master_detection_nomask_array == 1] = 0
        master_detection_nomask_array[master_detection_nomask_array > 0] = 1

        save_raster_template(tile_forest_mask, master_detection_masked_path,
                             master_detection_nomask_array, GDT_Byte, 0)

    #########################################################
    #########################################################
    #########################################################
    ### Get GFW data
    gfw_change_detection_folder = config.work_dir.joinpath("gfw_change_detection")
    os.makedirs(gfw_change_detection_folder, exist_ok=True)
    gfw_master_path = download_gfw_data(gfw_tile, sitecode, output_dir=str(gfw_change_detection_folder))


    gfw_tile_forest_mask = gfw_change_detection_folder.joinpath(f"{sitecode}_gfw_deforestation.tif")
    if not gfw_tile_forest_mask.exists():
        reproject_multibandraster_toextent(gfw_master_path, gfw_tile_forest_mask,
                                           epsg, pixel_width,
                                           xmin, imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax)

    reclassified_gfw_binarymask_path = gfw_change_detection_folder.joinpath(
        f"{sitecode}_gfw_post_2020_deforestation_binarymask.tif")
    if not reclassified_gfw_binarymask_path.exists():
        gfw_folder_tile_rasterdata = raster2array(gfw_tile_forest_mask)
        gfw_folder_tile_rasterdata[np.isnan(gfw_folder_tile_rasterdata)] = 0
        unique_data = np.unique(gfw_folder_tile_rasterdata)

        date_dict = decode_alert(unique_data)

        gfw_post_2021_deforestation_mask = np.zeros_like(gfw_folder_tile_rasterdata)
        for date_dict_keys, values in date_dict.items():
            if date_dict_keys < datetime.datetime(2021, 1, 1, 0, 0): continue
            for value_item in values:
                gfw_post_2021_deforestation_mask[gfw_folder_tile_rasterdata == value_item] = 1
        save_raster_template(gfw_tile_forest_mask, reclassified_gfw_binarymask_path, gfw_post_2021_deforestation_mask, GDT_Byte, 0)

    #########################################################
    #########################################################
    #########################################################
    # Forest degradation
    work_dir_3month_composite_folder = config.work_dir.joinpath(f"{sitecode}_3monthcomposite")
    os.makedirs(work_dir_3month_composite_folder, exist_ok=True)
    composite_start_timeperiod_master = datetime(2020, 1, 1)
    calculate_monthly_3month_composite(mosaic_file_pol_dict, composite_start_timeperiod_master,
                                       work_dir_3month_composite_folder)






if __name__ == "__main__":
    main()

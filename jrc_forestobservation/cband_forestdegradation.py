import os
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
from tempfile import mkdtemp
import shutil
import numpy as np
from osgeo.gdalconst import GDT_Float32, GDT_Int16
from src.tondor.util.tool import reproject_multibandraster_toextent, read_raster_info, raster2array, \
    save_raster_template
from src.tondor.util.raster import sieve_multiclass_tif
import rasterio
from rasterio.crs import CRS
from rasterio.shutil import copy as rio_copy

from tondortools.tool import save_raster_template

try:
    from COP4N2K_grassland_management.textures import run as make_texture
except:
    from internal.COP4N2K_grassland_mowing.textures import run as make_texture


def fifteenth_of_next_month(date):
    year = date.year
    month = date.month

    # Increment month, adjust year if needed
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1

    return date.replace(year=year, month=month, day=15)


def calculate_time_weight(target_date, reference_date):
    """
    Calculate a linear time weight based on proximity to the reference date.

    Parameters:
    - target_date (datetime): The date for which to calculate the weight.
    - reference_date (datetime): The central date with maximum weight 1.0.

    Returns:
    - float: Weight between 0.1 and 1.0 if within ±45 days, else 0.0
    """
    days_diff = abs((target_date - reference_date).days)

    if days_diff > 45:
        return 0.1
    else:
        # Linear interpolation from 1.0 (at day 0) to 0.1 (at day 45)
        return round(1.0 - (0.9 / 45) * days_diff, 4)


sarmosaic_folder = "/mnt/hddarchive.nfs/tmp/mosaic/ORBS"
composite_start_timeperiod_master = datetime(2020, 1, 15)
composite_end_timeperiod_master = datetime(2025, 2, 15)
orbit_to_study = 141
pol_to_study = "VV"
window_size = 3
work_dir = Path("/mnt/hddarchive.nfs/tmp/EUDR/test21LYG")
work_dir_clipped_folder = work_dir.joinpath("clipped_mosaic")

work_dir_3monthcomposite = work_dir.joinpath("3monthcomposite")
os.makedirs(work_dir_3monthcomposite, exist_ok=True)

work_dir_2yearcomposite = work_dir.joinpath("2yearcomposite")
os.makedirs(work_dir_2yearcomposite, exist_ok=True)

work_dir_2month_2year_diff = work_dir.joinpath("change_2month_2year_diff")
os.makedirs(work_dir_2month_2year_diff, exist_ok=True)

work_dir_2month_2year_diff_contrast = work_dir.joinpath(f"change_2month_2year_diff_contrast_win{window_size}")
os.makedirs(work_dir_2month_2year_diff_contrast, exist_ok=True)

mosaic_files = os.listdir(work_dir_clipped_folder)
# filter file with VV and VH polarisation in dictionary
mosaic_files = os.listdir(work_dir_clipped_folder)
mosaic_filepath = [Path(work_dir_clipped_folder).joinpath(mosaic_file_item) for mosaic_file_item in mosaic_files]
template_raster = mosaic_filepath[0]

mosaic_vv_filepath = [mosaic_file for mosaic_file in sorted(mosaic_filepath) if "_VV" in mosaic_file.name]
mosaic_vh_filepath = [mosaic_file for mosaic_file in sorted(mosaic_filepath) if "_VH" in mosaic_file.name]
print(f"mosaic_vv_filepath: {mosaic_vv_filepath}")
print(f"mosaic_vh_filepath: {mosaic_vh_filepath}")
mosaic_file_pol_dict = {}
mosaic_file_pol_dict["VV"] = mosaic_vv_filepath
mosaic_file_pol_dict["VH"] = mosaic_vh_filepath


def calculate_monthly_3month_composite(mosaic_file_pol_dict, composite_start_timeperiod_master,
                                     work_dir_3monthcomposite):
    ########
    # 3 month composite
    ########
    for pol, mosaic_filepath in mosaic_file_pol_dict.items():
        print(f"pol: {pol}")
        print(f"mosaic_filepath: {mosaic_filepath}")
        composite_start_timeperiod = composite_start_timeperiod_master
        while (composite_start_timeperiod < composite_end_timeperiod_master):
            # 3 month composite
            starttime_3monthcomposite = composite_start_timeperiod - timedelta(days=48)
            endtime_3monthcomposite = composite_start_timeperiod + timedelta(days=48)

            output_filepath = work_dir_3monthcomposite.joinpath(
                f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_3monthcomposite_{pol}.tif")
            if not output_filepath.exists():
                print(f"starttime_3monthcomposite:{composite_start_timeperiod} {starttime_3monthcomposite}")
                print(f"endtime_3monthcomposite:{composite_start_timeperiod} {endtime_3monthcomposite}")

                mosaic_pol_3month_list = []
                for mosaic_pol_filepath_item in mosaic_filepath:
                    mosaic_vv_date = datetime.strptime(mosaic_pol_filepath_item.name.split("_")[1], "%Y%m%d")
                    time_weight = calculate_time_weight(mosaic_vv_date, composite_start_timeperiod)
                    if starttime_3monthcomposite <= mosaic_vv_date <= endtime_3monthcomposite:
                        print(f"mosaic_vv_filepath_item: {mosaic_pol_filepath_item}")
                        mosaic_pol_3month_list.append((mosaic_pol_filepath_item, time_weight))
                    template_raster = mosaic_pol_filepath_item
                print(f"mosaic_pol_3month_list: {mosaic_pol_3month_list}")

                template_array = raster2array(template_raster)
                weighted_sum = np.zeros_like(template_array)
                weight_sum = 0.0
                for mosaic_pol_filepath_item, time_weight in mosaic_pol_3month_list:
                    mosaic_array = raster2array(mosaic_pol_filepath_item)
                    weighted_sum += mosaic_array * time_weight
                    weight_sum += time_weight
                weighted_average = weighted_sum / weight_sum
                save_raster_template(template_raster, work_dir_3monthcomposite.joinpath(
                    f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_3monthcomposite_{pol}.tif"), weighted_average,
                                     GDT_Float32)
            else:
                print(f"Output file {output_filepath} already exists, skipping.")
            composite_start_timeperiod = fifteenth_of_next_month(composite_start_timeperiod)

def calulcate_monthly_2yearpast_composite():
    ########
    # 2 year composite
    ########
    composite_file_pol_dict = {}
    for pol, mosaic_filepath in mosaic_file_pol_dict.items():
        print(f"pol: {pol}")
        print(f"mosaic_filepath: {mosaic_filepath}")
        composite_start_timeperiod = composite_start_timeperiod_master
        while (composite_start_timeperiod < composite_end_timeperiod_master):
            # 3 month composite
            composite_file_pol_dict[f"{composite_start_timeperiod.strftime('%Y%m%d')}"] = {}
            starttime_2year_composite = composite_start_timeperiod - timedelta(days=832)
            endtime_2year_composite = composite_start_timeperiod - timedelta(days=120)

            output_filepath = work_dir_2yearcomposite.joinpath(
                f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_{starttime_2year_composite.strftime('%Y%m%d')}_{endtime_2year_composite.strftime('%Y%m%d')}_2yearcomposite_{pol}.tif")
            if not output_filepath.exists():
                print(f"starttime_2year_composite:{composite_start_timeperiod} {starttime_2year_composite}")
                print(f"endtime_2year_composite:{composite_start_timeperiod} {endtime_2year_composite}")

                mosaic_pol_2year_list = []
                for mosaic_pol_filepath_item in mosaic_filepath:
                    mosaic_vv_date = datetime.strptime(mosaic_pol_filepath_item.name.split("_")[1], "%Y%m%d")
                    time_weight = calculate_time_weight(mosaic_vv_date, composite_start_timeperiod)
                    if starttime_2year_composite <= mosaic_vv_date <= endtime_2year_composite:
                        print(f"mosaic_vv_filepath_item: {mosaic_pol_filepath_item}")
                        mosaic_pol_2year_list.append((mosaic_pol_filepath_item, 1))
                    template_raster = mosaic_pol_filepath_item
                # print(f"mosaic_pol_2year_list: {mosaic_pol_2year_list}")

                template_array = raster2array(template_raster)
                weighted_sum = np.zeros_like(template_array)
                weight_sum = 0.0
                for mosaic_pol_filepath_item, time_weight in mosaic_pol_2year_list:
                    mosaic_array = raster2array(mosaic_pol_filepath_item)
                    weighted_sum += mosaic_array * time_weight
                    weight_sum += time_weight
                weighted_average = weighted_sum / weight_sum
                save_raster_template(template_raster, output_filepath, weighted_average, GDT_Float32)
            else:
                print(f"Output file {output_filepath} already exists, skipping.")
            composite_start_timeperiod = fifteenth_of_next_month(composite_start_timeperiod)

def calculate_3month_2year_composite(mosaic_file_pol_dict, composite_start_timeperiod_master, composite_start_timeperiod_master,
                                     number_of_days_to_start_2year_period, number_of_days_to_end_2year_period,
                                     work_dir_2yearcomposite):
    ########
    # composite difference
    ########
    for pol, mosaic_filepath in mosaic_file_pol_dict.items():
        print(f"pol: {pol}")
        print(f"mosaic_filepath: {mosaic_filepath}")
        composite_start_timeperiod = composite_start_timeperiod_master
        while (composite_start_timeperiod < composite_end_timeperiod_master):
            # 3 month composite
            composite_file_pol_dict[f"{composite_start_timeperiod.strftime('%Y%m%d')}"] = {}
            starttime_2year_composite = composite_start_timeperiod - timedelta(days=832)
            endtime_2year_composite = composite_start_timeperiod - timedelta(days=120)

            output_composite_2year_pol_filepath = work_dir_2yearcomposite.joinpath(
                f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_{starttime_2year_composite.strftime('%Y%m%d')}_{endtime_2year_composite.strftime('%Y%m%d')}_2yearcomposite_{pol}.tif")
            output_3month_composite_pol_filepath = work_dir_3monthcomposite.joinpath(
                f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_3monthcomposite_{pol}.tif")
            print(
                f"output_composite_2year_pol_filepath: {output_composite_2year_pol_filepath} - {output_3month_composite_pol_filepath}")
            diff_tifpath = work_dir_2month_2year_diff.joinpath(
                f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_3month_2year_diff_{pol}.tif")
            if diff_tifpath.exists():
                print(f"Output file {diff_tifpath} already exists, skipping.")
            else:
                array_3month = raster2array(output_3month_composite_pol_filepath)
                array_2year = raster2array(output_composite_2year_pol_filepath)
                diff_array = array_3month - array_2year
                save_raster_template(output_3month_composite_pol_filepath, diff_tifpath, diff_array, GDT_Float32)
            print(f"created {diff_tifpath}")
            composite_start_timeperiod = fifteenth_of_next_month(composite_start_timeperiod)

            asm_final_path = work_dir_2month_2year_diff_contrast.joinpath(f"{diff_tifpath.stem}_idm.tif")
            if not asm_final_path.exists():
                asm_dir = Path("/var/tmp").joinpath(diff_tifpath.stem)
                if asm_dir.exists(): shutil.rmtree(asm_dir)
                os.makedirs(asm_dir, exist_ok=True)
                shutil.copy(diff_tifpath, asm_dir.joinpath(f"{diff_tifpath.stem}_OPT.tif"))

                asm_grass_dir_parent = Path("/var/tmp").joinpath(f"{diff_tifpath.stem}_grass")
                os.makedirs(asm_grass_dir_parent, exist_ok=True)
                # launch grass session
                grass_base_dir = Path(mkdtemp(prefix="task_{:d}.".format(4444), suffix=".d", dir=asm_grass_dir_parent))
                grass_session_dir = grass_base_dir.joinpath("grass")
                grass_session_dir.mkdir(exist_ok=True, parents=True)
                print("GRASS session directory has been set to {:s}.".format(str(grass_session_dir)))

                asm_output_folder = Path("/var/tmp").joinpath(f"{diff_tifpath.stem}_output")
                os.makedirs(asm_output_folder, exist_ok=True)
                asm_path = asm_output_folder.joinpath(diff_tifpath.name.replace('.tif', '_OPT_idm.tif'))
                make_texture(str(asm_dir), ["idm"], 1, window_size=window_size, output_directory=str(asm_output_folder),
                             grass_session_folder=str(grass_session_dir))

                asm_final_path = work_dir_2month_2year_diff_contrast.joinpath(f"{diff_tifpath.stem}_idm.tif")

                # Define CRS and file paths
                crs = CRS.from_epsg(4326)
                src_path = str(asm_path)
                dst_path = str(asm_final_path)

                # Open the source raster
                with rasterio.open(src_path) as src:
                    # Copy the dataset and update the CRS
                    rio_copy(src, dst_path, copy_src_overviews=True, dst_crs=crs)

                # cmd_gdal = ["gdal_translate",
                #             "-a_srs", "EPSG:{}".format(4326),
                #             str(asm_path), str(asm_final_path)]
                # cmd_output = subprocess.run(cmd_gdal, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # print("exit code {} --> {}".format(cmd_output.returncode, cmd_gdal))

                asm_path.unlink()
                shutil.rmtree(grass_base_dir)
                shutil.rmtree(asm_dir)

template_array = raster2array(template_raster)
idm_array = np.zeros_like(template_array)

final_mask_list = []
for pol, mosaic_filepath in mosaic_file_pol_dict.items():
    print(f"pol: {pol}")
    print(f"mosaic_filepath: {mosaic_filepath}")
    binary_stack = []

    composite_start_timeperiod = composite_start_timeperiod_master
    while (composite_start_timeperiod < composite_end_timeperiod_master):
        composite_start_timeperiod = fifteenth_of_next_month(composite_start_timeperiod)
        # 3 month composite
        diff_tifpath = work_dir_2month_2year_diff.joinpath(
            f"BAC_{composite_start_timeperiod.strftime('%Y%m%d')}_3month_2year_diff_{pol}.tif")
        if diff_tifpath.exists():
            asm_final_path = work_dir_2month_2year_diff_contrast.joinpath(f"{diff_tifpath.stem}_idm.tif")
            idm_array += raster2array(asm_final_path)
            binary_stack.append(raster2array(asm_final_path) < .1)
    idm_sum_path = work_dir.joinpath(f"idm_{pol}_win{window_size}.tif")
    if not idm_sum_path.exists():
        save_raster_template(asm_final_path, idm_sum_path, idm_array, GDT_Float32)

    # Step 3: Detect three consecutive True values along time axis
    final_mask = np.zeros_like(template_array).astype(bool)
    if len(binary_stack) >= 3:
        for i in range(len(binary_stack) - 2):
            three_consecutive = (
                    binary_stack[i] & binary_stack[i + 1] & binary_stack[i + 2]
            )
            final_mask |= three_consecutive

    # Step 4: Convert boolean mask to float32 array
    final_mask = final_mask.astype(np.int16)
    idm_sum_path = work_dir.joinpath(f"final_mask_idm_{pol}_win{window_size}.tif")
    final_mask_list.append(idm_sum_path)
    if not idm_sum_path.exists():
        save_raster_template(template_raster, idm_sum_path, final_mask, GDT_Int16)

final_combined_mask = np.ones_like(template_array).astype(bool)
final_combine_path = work_dir.joinpath(f"final_combine_mask_idm_win{window_size}.tif")
if not final_combine_path.exists():
    for mask_path in final_mask_list:
        mask_array = raster2array(mask_path)
        final_combined_mask &= (mask_array > 0)
        save_raster_template(template_raster, final_combine_path, final_combined_mask, GDT_Int16)

final_combine_sieved_path = work_dir.joinpath(f"final_combine_mask_idm_win{window_size}_seived.tif")
if not final_combine_sieved_path.exists():
    sieve_multiclass_tif(final_combine_path, work_dir, final_combine_sieved_path)

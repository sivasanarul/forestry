import xarray as xr
import numpy as np
from scipy import stats
from skimage.morphology import remove_small_holes
from copy import copy
from openeo.udf.debug import inspect

DEBUG = False


def create_nan_mask(numpy_array, vv_vh_bandcount):
    # Create masks for both VV and VH parts of the array
    mask_vv = np.isnan(numpy_array[:vv_vh_bandcount]) | (numpy_array[:vv_vh_bandcount] < -99)
    mask_vh = np.isnan(numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount]) | (
            numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount] < -99)

    # Combine the masks (element-wise logical OR)
    combined_mask = mask_vv | mask_vh
    return combined_mask


def apply_threshold(stat_array, pol_item, DEC_array_threshold,
                    stat_item_name=None, previous_stat_array_bool=None):
    stat_array_copy = copy(stat_array)

    if stat_item_name == 'std':
        if pol_item == "VH":
            pol_thr = 2.0
        else:
            pol_thr = 1.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array < pol_thr), 0, 1)

    elif stat_item_name == 'mean_change':
        pol_thr = -1.75
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pol_thr), 0, 1)

    elif stat_item_name == 'tstat':
        tstat_thr = 3.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array < tstat_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'pval':
        pvalue_thr = 0.1
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pvalue_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'ratio_slope':
        stat_thr = 0.025
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_rsquared':
        stat_thr = 0.6
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_mean_change':
        stat_thr = 2.0
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_tstat':
        tstat_thr = 3.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array > tstat_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'ratio_pval':
        pvalue_thr = 0.1
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pvalue_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    DEC_array_threshold += stat_array.astype(int)
    if DEBUG:
        return DEC_array_threshold, stat_array_copy, stat_array.astype(int)
    else:
        return DEC_array_threshold, None, stat_array.astype(int)


def calculate_lsfit_r(vv_vh_r, vv_vh_bandcount):
    """
    Perform linear regression on band differences to calculate slope and R-squared.

    Parameters:
    vv_vh_r (np.array): Difference between VV and VH bands
    vv_vh_bandcount (int): Number of bands per polarization

    Returns:
    tuple: slope array, R-squared array
    """
    x = np.arange(vv_vh_bandcount)
    A = np.c_[x, np.ones_like(x)]

    y = np.reshape(vv_vh_r, (vv_vh_bandcount, -1))

    col_mean = np.nanmean(y, axis=0)
    inds = np.where(np.isnan(y))
    y[inds] = np.take(col_mean, inds[1])

    np.nan_to_num(y, copy=False, nan=0.0)

    m, resid, rank, s = np.linalg.lstsq(A, y, rcond=None)

    slope_r = np.reshape(m[0], vv_vh_r.shape[1:])
    r_squared = 1 - resid / (x.size * np.var(y, axis=0))
    r_squared = np.reshape(r_squared, vv_vh_r.shape[1:])

    return slope_r, r_squared


def apply_datacube(numpy_stack_pol_dict, window_size=10):

    half_window = window_size/2
    bands, dim1, dim2 = numpy_stack_pol_dict["VV"].shape
    DEC_array_threshold = np.zeros((dim1, dim2), dtype=int)

    for pol_item in ["VV", "VH"]:

        numpy_stack_pol = numpy_stack_pol_dict[pol_item]
        numpy_stack_past = numpy_stack_pol[0:5, :, :]
        numpy_stack_future = numpy_stack_pol[5:10, :, :]

        # Calculate the mean for Stack_p along the band axis (axis=0)
        Stack_p_MIN = np.nanmean(numpy_stack_past, axis=0)

        # Calculate the mean for Stack_f along the band axis (axis=0)
        Stack_f_MIN = np.nanmean(numpy_stack_future, axis=0)

        POL_std = np.nanstd(numpy_stack_pol, axis=0)
        DEC_array_threshold, POL_std, _ = apply_threshold(POL_std, pol_item, DEC_array_threshold,
                                                          stat_item_name="std")
        if not DEBUG: del POL_std

        ######## MOVING WINDOW TTEST ON STACK
        POL_mean_change = np.subtract(Stack_f_MIN, Stack_p_MIN)
        DEC_array_threshold, POL_mean_change, POL_mean_change_bool = apply_threshold(POL_mean_change, pol_item,
                                                                                     DEC_array_threshold,
                                                                                     stat_item_name="mean_change")

        # Perform t-test across bands
        # ttest_results = stats.ttest_ind(numpy_stack_past,
        #                                 numpy_stack_future,
        #                                 axis=0, nan_policy='omit')
        # ttest_pvalue, ttest_tstatistic = ttest_results.pvalue, ttest_results.statistic
        # DEC_array_threshold, ttest_pvalue, _ = apply_threshold(ttest_pvalue, pol_item, DEC_array_threshold,
        #                                                        stat_item_name="pval",
        #                                                        previous_stat_array_bool=POL_mean_change_bool)
        # DEC_array_threshold, ttest_tstatistic, _ = apply_threshold(ttest_tstatistic, pol_item, DEC_array_threshold,
        #                                                            stat_item_name="tstat",
        #                                                            previous_stat_array_bool=POL_mean_change_bool)

    numpy_stack_vv = numpy_stack_pol_dict["VV"]
    numpy_stack_vh = numpy_stack_pol_dict["VH"]
    ## VV - VH
    # vv_vh_r = np.subtract(numpy_array[list(np.arange(vv_vh_bandcount))] -
    #                       numpy_array[list(np.arange(vv_vh_bandcount) + vv_vh_bandcount)])
    vv_vh_r =numpy_stack_vv - numpy_stack_vh

    ratio_slope, ratio_r_squared = calculate_lsfit_r(vv_vh_r, window_size)
    DEC_array_threshold, ratio_slope, _ = apply_threshold(ratio_slope, pol_item, DEC_array_threshold,
                                                          stat_item_name="ratio_slope")
    DEC_array_threshold, ratio_r_squared, _ = apply_threshold(ratio_r_squared, pol_item, DEC_array_threshold,
                                                              stat_item_name="ratio_rsquared")

    if not DEBUG:
        del ratio_slope, ratio_r_squared

    # Calculate Mean Change Between Future and Past Stacks
    ratio_mean_change = np.nanmean(vv_vh_r[5:10, :, :]) - np.nanmean(vv_vh_r[0:5, :, :])


    DEC_array_threshold, ratio_mean_change, ratio_mean_change_bool = apply_threshold(ratio_mean_change, pol_item,
                                                                                     DEC_array_threshold,
                                                                                     stat_item_name="ratio_mean_change")

    if not DEBUG:
        del ratio_mean_change

    # Perform T-Test Efficiently
    ratio_ttest_results = stats.ttest_ind(vv_vh_r[0:5, :, :], vv_vh_r[5:10, :, :],
                                          axis=0, nan_policy='omit')

    # Extract p-value and t-statistic from the result
    DEC_array_threshold, ratio_ttest_pvalue, _ = apply_threshold(ratio_ttest_results.pvalue, pol_item,
                                                                 DEC_array_threshold,
                                                                 stat_item_name="ratio_pval",
                                                                 previous_stat_array_bool=ratio_mean_change_bool)
    DEC_array_threshold, ratio_ttest_tstatistic, _ = apply_threshold(ratio_ttest_results.statistic, pol_item,
                                                                     DEC_array_threshold,
                                                                     stat_item_name="ratio_tstat",
                                                                     previous_stat_array_bool=ratio_mean_change_bool)



    DEC_array_mask = np.zeros_like(DEC_array_threshold)
    DEC_array_mask[DEC_array_threshold > 3] = 1

    # Convert to boolean array (assuming the input is binary, 0 and 1)
    # Invert binary band
    inverted_band = np.logical_not(DEC_array_mask.astype(bool))
    processed_band = remove_small_holes(inverted_band, area_threshold=15)
    # Invert back to original
    DEC_array_mask = np.logical_not(processed_band).astype(np.uint8)

    return DEC_array_threshold, DEC_array_mask
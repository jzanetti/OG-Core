
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from matplotlib.pyplot import pcolor, savefig, subplots_adjust, subplot, colorbar, title, suptitle, close, plot
from os.path import exists
from os import makedirs
from numpy import asarray as np_asarray

def update_params(p):
    
    S = 8
    T = 30

    tG1 = 7
    tG2 = 15
    J = 7
    PSEUDO_DIM1 = 12
    PSEUDO_DIM2 = 1

    og_spec = {"S": S, "T": T, "tG1": tG1, "tG2": tG2, "J": J, "PSEUDO_DIM1": PSEUDO_DIM1, "PSEUDO_DIM2": PSEUDO_DIM2}

    return og_spec

def interpolate_partial(x, dim_i, start_index, new_tail_size, kind="linear"):
    """
    Interpolate an n-D array only beyond a certain index along one dimension.

    Parameters
    ----------
    x : np.ndarray
        Input array (e.g., shape (400, 3, 5))
    dim_i : int
        Axis to interpolate along (e.g., 0 for time)
    start_index : int
        Index where interpolation begins. Elements [0:start_index] are unchanged.
    new_tail_size : int
        Desired size of the interpolated tail beyond start_index.
    kind : str
        Interpolation type, default 'linear'

    Returns
    -------
    np.ndarray
        New array with shape identical to x except along dim_i:
        new_length = start_index + new_tail_size
    """
    x = np.asarray(x)
    old_len = x.shape[dim_i]

    # --- 1️⃣ Split only along dim_i ---
    prefix = np.take(x, indices=np.arange(start_index), axis=dim_i)
    tail   = np.take(x, indices=np.arange(start_index, old_len), axis=dim_i)

    old_tail_len = tail.shape[0] if dim_i == 0 else tail.shape[dim_i - 0]  # same regardless

    # --- 2️⃣ Build normalized grids (0–1 range) ---
    old_grid = np.linspace(0, 1, old_tail_len)
    new_grid = np.linspace(0, 1, new_tail_size)

    # --- 3️⃣ Interpolate only along that axis ---
    tail_moved = np.moveaxis(tail, dim_i, 0)  # now axis 0 = interpolation axis
    f = interp1d(old_grid, tail_moved, axis=0, kind=kind, fill_value="extrapolate")
    tail_new_moved = f(new_grid)
    tail_new = np.moveaxis(tail_new_moved, 0, dim_i)

    # --- 4️⃣ Concatenate prefix and new tail along the same axis ---
    x_new = np.concatenate([prefix, tail_new], axis=dim_i)
    return x_new



def interpolate_array(arr, new_shape, kind='linear'):
    """
    Interpolates a NumPy array to a new shape along each axis (except batch axes).
    
    Args:
        arr (np.ndarray): Input array (e.g., (320, 80) or (320, 80, 7))
        new_shape (tuple): Target shape (must have same number of dims)
        kind (str): Interpolation type ('linear', 'nearest', 'cubic', etc.)

    Returns:
        np.ndarray: Interpolated array with shape = new_shape
    """
    if arr.ndim != len(new_shape):
        raise ValueError("new_shape must have the same number of dimensions as arr")
    
    result = arr.copy()
    for axis, (old_size, new_size) in enumerate(zip(arr.shape, new_shape)):
        if old_size == new_size:
            continue  # skip unchanged dimensions
        old_x = np.linspace(0, 1, old_size)
        new_x = np.linspace(0, 1, new_size)
        f = interp1d(old_x, result, axis=axis, kind=kind)
        result = f(new_x)
    return result


def resample_method1(
        data_full, 
        og_spec,
        adjustments,
        plot_cfg = {"run_plot": False, "param_name": None, "savedir": ""}):
    """
    Downsample OG-Core omega (T_full, S_full) to smaller grid with optional buffer.
    If add_buffer=True, output shape will be (new_T + new_S, new_S),
    mimicking OG-Core's (T+S, S) convention.
    """

    orig_data_type = "array"
    if type(data_full) == list:
        data_full = np_asarray(data_full)
        orig_data_type = "list"

    if "T" in adjustments or "T+S" in adjustments:

        if "T+S" in adjustments:
            TS_sampled = og_spec["T"] + og_spec["S"]
        else:
            TS_sampled = og_spec["T"]

        if data_full.ndim == 1:
            data_resampled = data_full[:TS_sampled,]
            data_resampled_output = interpolate_array(data_resampled, (TS_sampled,))
        else:
            data_resampled = data_full[:TS_sampled, :]
            if len(adjustments) == 2:
                data_resampled_output = interpolate_array(data_resampled, (TS_sampled, og_spec[adjustments[1]]))
            elif len(adjustments) == 3:
                data_resampled_output = interpolate_array(
                    data_resampled, 
                    (TS_sampled, og_spec[adjustments[1]], og_spec[adjustments[2]]))
            else:
                raise Exception("Dimension error")

    else:
        data_resampled = data_full

        if  data_resampled.ndim == 1 and len(adjustments) == 1: 
            data_resampled_output = interpolate_array(data_full, (og_spec[adjustments[0]],))
        elif data_resampled.ndim == 2 and len(adjustments) == 2:
            data_resampled_output = interpolate_array(data_full, (og_spec[adjustments[0]], og_spec[adjustments[1]]))
        else:
            raise Exception("Dimension error")

    if plot_cfg["run_plot"]:

        if not exists(plot_cfg['savedir']):
            makedirs(plot_cfg['savedir'])

        if data_resampled.ndim == 3:
            data_full_to_plot = data_full[:, :, 0]
            data_resampled_output_to_plot = data_resampled_output[:, :, 0]
        else:
            data_full_to_plot = data_full
            data_resampled_output_to_plot = data_resampled_output

        if data_full_to_plot.ndim == 1:
            func_to_use = plot
            use_colorbar = False
        else:
            func_to_use = pcolor
            use_colorbar = True

        subplot(121)
        func_to_use(data_full_to_plot)
        if use_colorbar:
            colorbar(fraction=0.046)
        title(f"Full: min: {np.round(np.min(data_full), 2)}; max: {np.round(np.max(data_full), 2)}; \n mean: {np.round(np.mean(data_full), 2)}; total: {np.round(np.sum(data_full), 2)}")
        subplot(122)
        func_to_use(data_resampled_output_to_plot)
        if use_colorbar:
            colorbar(fraction=0.046,)
        title(f"Resampled: min: {np.round(np.min(data_resampled), 2)}; max: {np.round(np.max(data_resampled), 2)}; \n mean: {np.round(np.mean(data_resampled), 2)}; total: {np.round(np.sum(data_full), 2)}")
        suptitle(f"{plot_cfg['param_name']}", y=1.05)
        subplots_adjust(wspace=0.4) 
        savefig(f"{plot_cfg['savedir']}/{plot_cfg['param_name']}.png", bbox_inches='tight')
        close()


    if orig_data_type == "list":
        data_resampled_output = data_resampled_output.tolist()

    return data_resampled_output


def resample_ts2(
        data_full, 
        new_T, 
        new_S, 
        add_buffer=True, 
        method="linear",
        apply_scaler = False,
        plot_cfg = {"run_plot": False, "param_name": None, "savedir": ""}):
    """
    Downsample OG-Core omega (T_full, S_full) to smaller grid with optional buffer.
    If add_buffer=True, output shape will be (new_T + new_S, new_S),
    mimicking OG-Core's (T+S, S) convention.
    """

    T_full, S_full = data_full.shape

    data_full = data_full[0: new_T + new_S , :]

    data_resampled = []
    for i in range(new_T + new_S):
        data_resampled.append(interpolate_partial(data_full[i, :], 0, 0, new_S, kind=method))

    data_resampled = np_asarray(data_resampled)

    if plot_cfg["run_plot"]:

        if not exists(plot_cfg['savedir']):
            makedirs(plot_cfg['savedir'])

        subplot(121)
        pcolor(data_full)
        colorbar(fraction=0.046)
        title(f"Full: min: {np.round(np.min(data_full), 2)}; max: {np.round(np.max(data_full), 2)}; \n mean: {np.round(np.mean(data_full), 2)}; total: {np.round(np.sum(data_full), 2)}")
        subplot(122)
        pcolor(data_resampled)
        colorbar(fraction=0.046,)
        title(f"Resampled: min: {np.round(np.min(data_resampled), 2)}; max: {np.round(np.max(data_resampled), 2)}; \n mean: {np.round(np.mean(data_resampled), 2)}; total: {np.round(np.sum(data_full), 2)}")
        suptitle(f"{plot_cfg['param_name']}", y=1.05)
        subplots_adjust(wspace=0.4) 
        savefig(f"{plot_cfg['savedir']}/{plot_cfg['param_name']}.png", bbox_inches='tight')
        close()


    return data_resampled

    # target number of time points
    target_T = new_T + new_S if add_buffer else new_T

    # coordinate grids
    t_full = np.linspace(0, T_full - 1, T_full)
    s_full = np.linspace(0, S_full - 1, S_full)
    t_new = np.linspace(0, T_full - 1, target_T)
    s_new = np.linspace(0, S_full - 1, new_S)

    # interpolator
    interp = RegularGridInterpolator(
        (t_full, s_full), data_full, method=method,
        bounds_error=False, fill_value=None
    )

    # make coordinate mesh for new grid
    tt, ss = np.meshgrid(t_new, s_new, indexing="ij")
    points_new = np.column_stack([tt.ravel(), ss.ravel()])

    data_resampled = interp(points_new).reshape(target_T, new_S)

    # renormalize each year to sum to 1
    if data_resampled.sum() != 0 and apply_scaler:
        data_resampled /= data_resampled.sum(axis=1, keepdims=True)
    
    if plot_cfg["run_plot"]:

        if not exists(plot_cfg['savedir']):
            makedirs(plot_cfg['savedir'])

        subplot(121)
        pcolor(data_full)
        colorbar(fraction=0.046)
        title(f"Full: min: {np.round(np.min(data_full), 2)}; max: {np.round(np.max(data_full), 2)}; \n mean: {np.round(np.mean(data_full), 2)}; total: {np.round(np.sum(data_full), 2)}")
        subplot(122)
        pcolor(data_resampled)
        colorbar(fraction=0.046,)
        title(f"Resampled: min: {np.round(np.min(data_resampled), 2)}; max: {np.round(np.max(data_resampled), 2)}; \n mean: {np.round(np.mean(data_resampled), 2)}; total: {np.round(np.sum(data_full), 2)}")
        suptitle(f"{plot_cfg['param_name']}", y=1.05)
        subplots_adjust(wspace=0.4) 
        savefig(f"{plot_cfg['savedir']}/{plot_cfg['param_name']}.png", bbox_inches='tight')
        close()

    return data_resampled


def resample_s(
        data_full, 
        new_S, 
        method="linear",
        apply_scaler = False,
        plot_cfg = {"run_plot": False, "param_name": None, "savedir": ""}):
    

    data_resampled = interpolate_partial(data_full, 0, 0, new_S, kind=method)

    if apply_scaler:
        data_resampled = data_resampled * (1.0 / data_resampled.sum())

    if plot_cfg["run_plot"]:

        if not exists(plot_cfg['savedir']):
            makedirs(plot_cfg['savedir'])

        subplot(121)
        plot(data_full)
        colorbar(fraction=0.046)
        title(f"Full: min: {np.round(np.min(data_full), 2)}; max: {np.round(np.max(data_full), 2)}; \n mean: {np.round(np.mean(data_full), 2)}; total: {np.round(np.sum(data_full), 2)}")
        subplot(122)
        plot(data_resampled)
        colorbar(fraction=0.046,)
        title(f"Resampled: min: {np.round(np.min(data_resampled), 2)}; max: {np.round(np.max(data_resampled), 2)}; \n mean: {np.round(np.mean(data_resampled), 2)}; total: {np.round(np.sum(data_full), 2)}")
        suptitle(f"{plot_cfg['param_name']}", y=1.05)
        subplots_adjust(wspace=0.4) 
        savefig(f"{plot_cfg['savedir']}/{plot_cfg['param_name']}.png", bbox_inches='tight')
        close()
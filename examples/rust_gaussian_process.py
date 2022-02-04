import ctypes
from ctypes import c_double, c_int64, c_size_t, c_float
import numpy as np
from time import perf_counter

lib_path = "target/release/rusty_gaussian_processes.dll"

gp_lib = ctypes.cdll.LoadLibrary(lib_path)

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

gp_lib.rust_single_gp.argtypes = (
    ND_POINTER_1,
    ND_POINTER_1,
    c_size_t,
    ND_POINTER_1,
    c_size_t,
    c_int64,
    c_int64,
    c_double,
    c_double,
    c_double,
)

gp_lib.rust_single_gp.restype = None


def rust_run_gp(
    x_input,
    y_input,
    forecast_spacing,
    forecast_amount,
    length_scale=30,
    amplitude=0.5,
    noise=0.1,
):

    result = np.empty(x_input.size + forecast_amount)

    if not x_input.flags["C_CONTIGUOUS"]:
        x_input = np.ascontiguousarray(x_input)

    if not y_input.flags["C_CONTIGUOUS"]:
        y_input = np.ascontiguousarray(y_input)

    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)

    gp_lib.rust_single_gp(
        x_input,
        y_input - np.mean(y_input),
        y_input.size,
        result,
        result.size,
        c_int64(forecast_spacing),
        c_int64(forecast_amount),
        c_double(length_scale),
        c_double(amplitude),
        c_double(noise),
    )

    return result + np.mean(y_input)

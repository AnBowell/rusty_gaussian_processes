import ctypes
from ctypes import c_double, c_int64, c_size_t, c_float
import numpy as np
from time import perf_counter

lib_path = "target/release/rusty_gaussian_processes.dll"

gp_lib = ctypes.cdll.LoadLibrary(lib_path)


float_pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
usize_pointer = np.ctypeslib.ndpointer(dtype=c_size_t, ndim=1, flags="C")


gp_lib.rust_single_gp.argtypes = (
    float_pointer,
    float_pointer,
    c_size_t,
    float_pointer,
    c_size_t,
    c_int64,
    c_int64,
    c_double,
    c_double,
    c_double,
)

gp_lib.rust_single_gp.restype = None


gp_lib.rust_multiple_gps.argtypes = (
    float_pointer,
    float_pointer,
    c_size_t,
    usize_pointer,
    c_size_t,
    float_pointer,
    c_size_t,
    c_int64,
    c_int64,
    c_double,
    c_double,
    c_double,
)

gp_lib.rust_multiple_gps.restype = None


def rust_run_single_gp(
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


def run_multiple_gps(
    x_inputs,
    y_inputs,
    forecast_spacing,
    forecast_amount,
    length_scale=30,
    amplitude=0.5,
    noise=0.1,
):

    number_of_inputs = len(x_inputs)

    index_runner = 0

    start_indices = [0]

    for x_input in x_inputs[:-1]:
        length_of_input = len(x_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_input_array = np.array(y_inputs, dtype=np.float64)

    x_input_array = np.concatenate(x_inputs).ravel().astype(np.float64)

    y_input_means = np.mean(y_input_array, axis=1, keepdims=True)

    y_input_array = (y_input_array - y_input_means).ravel()

    result = np.empty(
        x_input_array.size + (forecast_amount * number_of_inputs), dtype=np.float64
    )

    if not x_input_array.flags["C_CONTIGUOUS"]:
        x_input_array = np.ascontiguousarray(x_input_array)

    if not y_input_array.flags["C_CONTIGUOUS"]:
        y_input_array = np.ascontiguousarray(y_input_array)

    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)

    gp_lib.rust_multiple_gps(
        x_input_array,
        y_input_array,
        x_input_array.size,
        start_indices,
        start_indices.size,
        result,
        result.size,
        c_int64(forecast_spacing),
        c_int64(forecast_amount),
        c_double(length_scale),
        c_double(amplitude),
        c_double(noise),
    )

    results = []

    start_indices[1:] += (
        np.arange(1, len(start_indices[1:]) + 1) * forecast_amount
    ).astype(np.uint64)

    for i in range(0, len(start_indices)):

        if i + 1 >= len(start_indices):

            single_result = result[start_indices[int(i)] :] + y_input_means[i]
        else:

            single_result = (
                result[start_indices[int(i)] : int(start_indices[i + 1])]
                + y_input_means[i]
            )

        results.append(single_result)

    return results

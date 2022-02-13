import ctypes
from ctypes import c_double, c_int64, c_size_t
import numpy as np

# The code below loads the dll and sets the input/output types.
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
    length_scale=50.0,
    amplitude=0.5,
    noise=0.01,
):
    """Wrapper function to run a single GP through the Rust library.
    Simply pass the x and y you want to fit, a forecast of spacing and amount,
    and this wrapper will call the Rust function. The y input in this case
    should not have the mean removed and any Nans/infs etc should be removed
    from the dataset.


    Args:
        x_input ([float64]): The x-axis input. For VCI forcasting, time in days.
        y_input ([float64]): The y-axis input. For VCI forecasting, the VCI.
        forecast_spacing (int): The spacing between the forecast. For weekly, 7.
        forecast_amount (int): The amount of forecasts. 10 would yield 10 forecasts of forecasting_spacing.
        length_scale (float, optional): Lengthscale of the squared-exp Kernel. Defaults to 50.
        amplitude (float, optional): Amplitude of the squared-exp Kernel. Defaults to 0.5.
        noise (float, optional): Noise of the GP regression. Defaults to 0.01.

    Returns:
        [float64]: The result of the GP regression sampled at each input as well as
        the requested forecasts.
    """

    result = np.empty(x_input.size + forecast_amount)

    # If data is not contiguous, using sending a pointer of the NumPy arrays
    # to the Rust library will not work! So good to check.
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
    y_inputs_mean_removed,
    forecast_spacing,
    forecast_amount,
    length_scale=30,
    amplitude=0.5,
    noise=0.1,
):
    """Wrapper function to run multiple GPs multithreaded through the Rust library.

    This function is a little more complex compared to the single GP function.
    The inputs/outputs should be a list of numpy arrays that you want to forecast.
    All of the arrays are then combined and the indicies of where each one starts
    is saved. This is then all passed to the rust function and a train/forecast
    cycle is performed on each one. This wrapper function then unpacks the results
    back into a list of arrays.

    Note: The y inputs should have their mean removed. This is handled in the
    single GP function, but here, it is more effiecent for the user to do it.

    Args:
        x_inputs ([[float]]): A list of numpy arrays containing the x-axis input.
        y_inputs_mean_removed ([[float]]): A list of numpy arrays containing the y-axis input.
            These arrays should all have a mean of zero! Remove the mean before calling it.
        forecast_spacing (int): The spacing between the forecast. For weekly, 7.
        forecast_amount (int): The amount of forecasts. 10 would yield 10 forecasts of forecasting_spacing.
        length_scale (float, optional): Lengthscale of the squared-exp Kernel. Defaults to 50.
        amplitude (float, optional): Amplitude of the squared-exp Kernel. Defaults to 0.5.
        noise (float, optional): Noise of the GP regression. Defaults to 0.01.

    Returns:
        [[float64]]: A list of arrays containing the results of each GP forecast.
    """

    number_of_inputs = len(x_inputs)

    index_runner = 0

    start_indices = [0]

    for x_input in x_inputs[:-1]:
        length_of_input = len(x_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_input_array = np.concatenate(y_inputs_mean_removed).ravel().astype(np.float64)
    x_input_array = np.concatenate(x_inputs).ravel().astype(np.float64)

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

            single_result = result[start_indices[int(i)] :]
        else:

            single_result = result[start_indices[int(i)] : int(start_indices[i + 1])]

        results.append(single_result)

    return results

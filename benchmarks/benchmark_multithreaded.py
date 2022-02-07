import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import h5py as h5
import numpy as np
from gaussian_processes import rust_gaussian_process, gaussian_process
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import perf_counter


data_file = h5.File("data/FinalSubCountyVCI.h5", "r+")


test_key = list(data_file.keys())[0]


dataset = data_file[test_key]
dates = dataset[:, 0][:1000]
VCI3M = dataset[:, 3][:1000]


nan_mask = np.isnan(VCI3M)
dates, VCI3M = dates[~nan_mask], VCI3M[~nan_mask]
zeros_mask = dates != 0
dates, VCI3M = dates[zeros_mask], VCI3M[zeros_mask]

VCI3M = VCI3M.astype(np.float64)

dates = np.array(
    [
        datetime(int(str(date)[:4]), 1, 1) + timedelta(int(str(date)[4:7]) - 1)
        for date in dates
    ]
)
days = np.array([(date - dates[0]).days for date in dates], dtype=np.float64)


rust_multiple_times, rust_single_times, python_times = [], [], []

amounts_to_test = [1, 5, 10, 20]

for amount_of_runs in amounts_to_test:

    x_inputs = [days] * amount_of_runs
    y_inputs = [VCI3M] * amount_of_runs
    y_inputs_mean_removed = [VCI3M - np.mean(VCI3M)] * amount_of_runs

    # ~~~~~~~~~~~~~~ Bench mark rust multithreaded.~~~~~~~~~~~~~~~~~~~#

    start_time = perf_counter()

    results = rust_gaussian_process.run_multiple_gps(
        x_inputs,
        y_inputs_mean_removed,
        forecast_spacing=7,
        forecast_amount=10,
        length_scale=25,
        amplitude=2,
        noise=0.5,
    )

    end_time = perf_counter()

    # print("Time take to process = {}s".format(end_time - start_time))

    rust_multiple_gp_time = end_time - start_time
    rust_multiple_times.append(rust_multiple_gp_time)
    # ~~~~~~~~~~~~~~~~~~~~~ Benchmark rust single threaded ~~~~~~~~~~~#

    start_time = perf_counter()

    results = [
        rust_gaussian_process.rust_run_single_gp(
            x_input,
            y_input,
            forecast_spacing=7,
            forecast_amount=10,
            length_scale=25,
            amplitude=2,
            noise=0.5,
        )
        for x_input, y_input in zip(x_inputs, y_inputs)
    ]

    end_time = perf_counter()

    rust_single_thread_time = end_time - start_time
    rust_single_times.append(rust_single_thread_time)
    # print("Time take to process = {}s".format(end_time - start_time))

    # ~~~~~~~~~~~~~~~~~~ Benchmark Python ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    start_time = perf_counter()

    results = [
        gaussian_process.forecast(x_input, y_input)
        for x_input, y_input in zip(x_inputs, y_inputs)
    ]

    end_time = perf_counter()

    python_time = end_time - start_time
    # print("Time take to process = {}s".format(end_time - start_time))
    python_times.append(python_time)


fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    amounts_to_test,
    rust_multiple_times,
    color="red",
    label="Rust multithreaded",
    marker="x",
)
ax1.plot(
    amounts_to_test,
    rust_single_times,
    color="blue",
    label="Rust singlethreaded",
    marker="x",
)

ax1.plot(
    amounts_to_test,
    python_times,
    color="green",
    label="Python (pyro)",
    marker="x",
)

ax1.grid(True, alpha=0.6, ls="--")


plt.show()
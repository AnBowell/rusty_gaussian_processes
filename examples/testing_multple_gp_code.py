import numpy as np
import gaussian_processes.rust_gaussian_process as rust_gaussian_process
from time import perf_counter
import pandas as pd

data = pd.read_csv("data/proc_test_vci3m.csv")

days_since = data["dates"].to_numpy(dtype=np.float64)
vci = data["VCI3M"].to_numpy(dtype=np.float64)


forecast_spacing = 7
forecast_amount = 10

test_start = 0
test_end = len(vci)

x_inputs = [days_since[test_start:test_end]] * 100
y_inputs = [vci[test_start:test_end]] * 100


start_time = perf_counter()

results = rust_gaussian_process.run_multiple_gps(
    x_inputs,
    y_inputs,
    forecast_spacing=forecast_spacing,
    forecast_amount=forecast_amount,
    length_scale=30,
    amplitude=2.0,
)

end_time = perf_counter()

print("Rust multithreaded GP took {}s".format(end_time - start_time))

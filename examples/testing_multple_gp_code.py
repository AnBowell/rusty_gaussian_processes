from os import X_OK
import h5py as h5
import numpy as np
import gaussian_process
import rust_gaussian_process
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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


# for result in results:
#     print(len(result))
#     plt.plot(
#         days_since[test_start : test_end + 10],
#         vci[test_start : test_end + 10],
#         label="Raw",
#         color="green",
#     )
#     plt.plot(
#         np.concatenate(
#             [
#                 days_since[test_start:test_end],
#                 days_since[test_start:test_end][-1]
#                 + np.arange(1, forecast_amount + 1) * forecast_spacing,
#             ]
#         ),
#         result,
#         color="blue",
#         label="Forecast",
#     )
#     plt.legend()
#     plt.show()

import numpy as np
import pandas as pd
import gaussian_process
from time import perf_counter
import rust_gaussian_process
import matplotlib.pyplot as plt

data = pd.read_csv("data/proc_test_vci3m.csv")


days_since = data["dates"].to_numpy(dtype=np.float64)[100:700]
vci = data["VCI3M"].to_numpy(dtype=np.float64)[100:700]


pyro_start = perf_counter()

new_dates, pyro_smoothed_data = gaussian_process.forecast(days_since, vci)

pyro_end = perf_counter()

print("Pyro GP done. This took {}s".format(pyro_end - pyro_start))


rust_start = perf_counter()

rust_smoothed_data = rust_gaussian_process.rust_run_gp(
    days_since, vci, forecast_spacing=7, forecast_amount=10, length_scale=40
)

rust_end = perf_counter()
print("Rust GP done. This took {}s".format(rust_end - rust_start))

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    new_dates,
    rust_smoothed_data,
    label="rust_smoothed",
    color="red",
)
ax1.plot(
    new_dates,
    pyro_smoothed_data,
    label="python",
    color="blue",
)
ax1.plot(
    days_since,
    vci,
    label="raw",
    color="green",
)
ax1.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()

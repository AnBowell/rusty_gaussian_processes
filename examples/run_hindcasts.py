from msilib.schema import File
import h5py as h5
import numpy as np
import gaussian_process
import rust_gaussian_process
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from time import perf_counter

data_file = h5.File("data/FinalSubCountyVCI.h5", "r+")

forecast_amount = 10


def main():
    for key_name in data_file.keys():

        dataset = data_file[key_name]

        dates = dataset[:, 0]
        VCI3M = dataset[:, 3]

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

        data_length = len(days)

        halfway_point = int(data_length / 2)

        hindcast_values = np.full(
            (data_length - halfway_point + forecast_amount, forecast_amount), 0.0
        )
        actual_values = np.full(
            (data_length - halfway_point + forecast_amount, forecast_amount), 0.0
        )

        for run_counter, hindcast_counter in enumerate(
            range(halfway_point, data_length)
        ):

            rust_start = perf_counter()

            vci3m_to_run = VCI3M[:hindcast_counter]
            days_to_run = days[:hindcast_counter]

            new_dates, rust_smoothed_data = gaussian_process.forecast(
                days_to_run, vci3m_to_run
            )
            # rust_smoothed_data = rust_gaussian_process.rust_run_gp(
            #     days_to_run,
            #     vci3m_to_run,
            #     forecast_spacing=7,
            #     forecast_amount=forecast_amount,
            #     length_scale=40,
            #     amplitude=2,
            # )

            rust_end = perf_counter()
            print("Rust GP done. This took {}s".format(rust_end - rust_start))

            actual_values[run_counter, :] = VCI3M[
                hindcast_counter : hindcast_counter + 10
            ]
            hindcast_values[run_counter] = rust_smoothed_data[
                hindcast_counter : hindcast_counter + 10
            ]

            print(np.abs(actual_values[run_counter, :] - hindcast_values[run_counter]))

            if run_counter % 25 == 0:
                print(
                    "{} out of {}\n\n\n".format(
                        run_counter, data_length - halfway_point
                    )
                )

        np.save("outputs/actual_vals.npy", actual_values)
        np.save("outputs/hindcast_vals_npy", hindcast_values)
        break


def plot_data(original_days, original, new_days, rust, python):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        new_days,
        rust,
        label="rust_smoothed",
        color="red",
    )
    ax1.plot(
        new_days,
        python,
        label="python",
        color="blue",
    )
    ax1.plot(
        original_days,
        original,
        label="raw",
        color="green",
    )
    ax1.grid(True, ls="--", alpha=0.6)
    plt.legend()


if __name__ == "__main__":
    main()

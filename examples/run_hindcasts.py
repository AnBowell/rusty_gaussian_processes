from msilib.schema import File
import h5py as h5
import numpy as np
import gaussian_processes.gaussian_process as gaussian_process
import gaussian_processes.rust_gaussian_process as rust_gaussian_process
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

        x_inputs = []
        y_inputs_mean_removed = []
        y_inputs_means = []
        actual_values = []
        for run_counter, hindcast_counter in enumerate(
            range(halfway_point, data_length - 10)
        ):

            rust_start = perf_counter()

            vci3m_to_run = VCI3M[run_counter:hindcast_counter]
            actual_values.append(VCI3M[hindcast_counter : hindcast_counter + 10])

            days_to_run = days[run_counter:hindcast_counter]

            x_inputs.append(days_to_run)
            mean = np.mean(vci3m_to_run)
            y_inputs_means.append(mean)
            y_inputs_mean_removed.append(vci3m_to_run - mean)

        results = rust_gaussian_process.run_multiple_gps(
            x_inputs,
            y_inputs_mean_removed,
            forecast_spacing=7,
            forecast_amount=10,
            length_scale=25,
            amplitude=2,
            noise=0.5,
        )

        final_results = []
        rmse = []

        for result, y_means, actual in zip(results, y_inputs_means, actual_values):

            # forecast_vci = np.full(10, -999)

            # for i in range(1, 11):

            #     forecast_vci[i - 1] = np.mean(result[(-12) + i :-10+1] + y_means)

            forecast_vci = result[-10:] + y_means

            final_results.append(abs(actual - forecast_vci))
            rmse.append(abs(actual - forecast_vci) ** 2)

        print("Dataset: {}".format(dataset))
        print(np.sqrt(np.mean(rmse, axis=0)))
        print(np.std(final_results, axis=0))
        # print(np.mean(final_results, axis=0))

        # break
        print("Done")

    print("Done")


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

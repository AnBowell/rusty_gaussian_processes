from re import sub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5


rmse_data = np.load("benchmarks/outputs/vci3m_rmse.npy")
rsquared_data = np.load("benchmarks/outputs/vci3m_r2.npy")

data_file = h5.File("data/FinalSubCountyVCI.h5", "r+")
datasets = list(data_file.keys())


col_names = ["Week {}".format(i) for i in range(1, 11)]


subcounty_rmse = pd.DataFrame(rmse_data, index=datasets, columns=col_names)
subcounty_rmse.loc["mean"] = subcounty_rmse.mean()
subcounty_rmse.to_csv("benchmarks/outputs/subcounty_rmses.csv")


subcounty_r2 = pd.DataFrame(rsquared_data, index=datasets, columns=col_names)
subcounty_r2.loc["mean"] = subcounty_r2.mean()
subcounty_r2.to_csv("benchmarks/outputs/subcounty_r2s.csv")


print(subcounty_rmse)
print(subcounty_r2)

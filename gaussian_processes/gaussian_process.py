# -*- coding: utf-8 -*-
"""This module houses the Gaussian processes prediction function

Using two machine learning Python modules, PyTorch and Pyro VCI3M is forecast
90 days into the future.

Created on Mon Nov 18 17:32:54 2019
@author: Andrew

"""
# Importing modules used
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

# Setting some global attribute settings

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)


def forecast(X, y):
    """The function that will create the forecasts

    Parameters
    ---------
    X : :obj:`Numpy array` of :obj:`float`
        Days since first datapoint. E.g 0,10,20,30,40
    y : :obj:`Numpy array` of :obj:`float`
        VCI3M recorded on the given day

    Returns
    -------
    Xtest_use : :obj:`Numpy array` of :obj:`float`
        Days since first datapoint
    mean : :obj:`Numpy array` of :obj:`float`
        This array includes the previous known VCI3M data but smoothed with
        the RBF kernel. It also includes the 90 days of forecasted VCI3M.

    """

    # Create RBF kernel using pyro and define lengthscale and variance.
    k1 = gp.kernels.RBF(
        input_dim=2, lengthscale=torch.tensor(50.0), variance=torch.tensor(0.5)
    )

    # Standard for VCI3M = 50 and 0.5

    optim = Adam({"lr": 0.01})

    pyro.clear_param_store()

    # Append the forecast length onto the known days. (9, 10 day forecasts)

    # plus_arr = np.max(X)+np.array([10.,20.,30.,40.,50.,60.,70.,80.,90.])
    plus_arr = np.max(X) + np.array(
        [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0]
    )
    Xtest_use = np.append(X, plus_arr)

    # Turn the numpy arrays into Torch tensors (for quicker calcualtions)

    X2 = torch.from_numpy(X)
    y2 = torch.from_numpy(y - np.mean(y))

    Xtest_use2 = torch.from_numpy(Xtest_use)

    # Activate the module using the data and the kernel. Specify the noise

    gpr = gp.models.GPRegression(X2, y2, k1, noise=torch.tensor(0.01))

    # Stochastic Variational Inference to find optimal fit for the data

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())

    # Specify how many steps of the SVI

    num_steps = 10
    losses = np.empty(num_steps)

    # Step through the SVI

    for k in range(num_steps):
        losses[k] = svi.step()

    # Convert back to numpy arrays and then return them.

    with torch.no_grad():
        if type(gpr) == gp.models.VariationalSparseGP:
            mean, cov = gpr(Xtest_use2, full_cov=True)
        else:
            mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False)

    mean = mean.detach().numpy() + np.mean(y)
    # sd = cov.sqrt().detach().numpy()
    # print(sd[-11:])
    return Xtest_use, mean

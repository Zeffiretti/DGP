import gpytorch


class MultioutputDGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, num_outputs):
    super(MultioutputDGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    # self.covar_module = gpytorch.kernels.RBFKernel()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # We learn an IndexKernel for 2 tasks
    # (so we'll actually learn 2x2=4 tasks with correlations)
    self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_outputs, rank=2)
    # self.task_covar_module = gpytorch.kernels.GridInterpolationKernel(self.covar_module, grid_size=1,
    # num_dims=points_number)

  def forward(self, x, i):
    mean_x = self.mean_module(x)

    # Get input-input covariance
    covar_x = self.covar_module(x)
    # Get task-task covariance
    covar_i = self.task_covar_module(i)
    # Multiply the two together to get the covariance we want
    covar = covar_x.mul(covar_i)

    return gpytorch.distributions.MultivariateNormal(mean_x, covar)

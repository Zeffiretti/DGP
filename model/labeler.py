###########################################################################
###                    Optic Data Auto-label Program                    ###
###                          ZEFFIRETTI, HIESH                          ###
###                   Beijing Institute of Technology                   ###
###                zeffiretti@bit.edu.cn, hiesh@mail.com                ###
###########################################################################

import torch
import gpytorch
from .modgp import MultioutputDGPModel


class Label:
  def __init__(self, init_time, init_data, device=torch.device('cpu'), max_iter=50):
    self.device = device
    assert init_data.shape[1] % 3 == 0, "init data shape error: some points may not contain 3-axis data"
    self.num_output = init_data.shape[1] // 3
    print('output num', self.num_output)
    self.train_datas = self.wrap_train_data(init_data)
    self.train_time, self.train_task = self.wrap_time(init_time, self.num_output)
    self.likelihoods = {'x': gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device),
                        'y': gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device),
                        'z': gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device)}
    self.models = {
      'x': MultioutputDGPModel((self.train_time, self.train_task), self.train_datas['x'], self.likelihoods['x'],
                               self.num_output).to(self.device),
      'y': MultioutputDGPModel((self.train_time, self.train_task), self.train_datas['y'], self.likelihoods['y'],
                               self.num_output).to(self.device),
      'z': MultioutputDGPModel((self.train_time, self.train_task), self.train_datas['z'], self.likelihoods['z'],
                               self.num_output).to(self.device)}
    self.optimizers = {'x': torch.optim.Adam([{'params': self.models['x'].parameters()}, ], lr=0.3),
                       'y': torch.optim.Adam([{'params': self.models['y'].parameters()}, ], lr=0.3),
                       'z': torch.optim.Adam([{'params': self.models['z'].parameters()}, ], lr=0.3),
                       }  # Includes GaussianLikelihood parameters
    self.mlls = {'x': gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods['x'], self.models['x']).to(device),
                 'y': gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods['y'], self.models['y']).to(device),
                 'z': gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods['z'], self.models['z']).to(device), }
    for axis in self.models:
      print(axis, ':', self.models[axis])
    # print(next(self.models[axis].parameters()).device)
    self.max_iter = max_iter

    # def init_model(self, train_x, train_y, likelihood, num_outputs):
    #   return MultioutputDGPModel(train_x, train_y, likelihood, num_outputs)

  def wrap_time(self, single_time, num_output):
    """
    tile single column time into time-task format
    :param single_time:
    :type single_time: torch.tensor, N x 1
    :param num_output:
    :type num_output: int
    :return:
    """
    times = single_time.repeat(num_output)
    tasks = torch.full_like(single_time, dtype=torch.long, fill_value=0).to(self.device)
    for i in range(1, num_output):
      task_i = torch.full_like(single_time, dtype=torch.long, fill_value=i).to(self.device)
      tasks = torch.cat([tasks, task_i])
    return times, tasks

  def wrap_train_data(self, all_data):
    """
    split data containing 3-axis features into each channel
    :param all_data:
    :type all_data: torch.tensor, rowCnt: N, colCnt: num_output x 3
    :return:
    """
    colCnt = all_data.shape[1]
    assert colCnt % 3 == 0, "data shape error: some points may not contain 3-axis data"
    rowCnt = all_data.shape[0]
    x_idx = list(filter(lambda x: x % 3 == 1, range(colCnt)))
    y_idx = list(filter(lambda y: y % 3 == 2, range(colCnt)))
    z_idx = list(filter(lambda z: z % 3 == 0, range(colCnt)))
    x_data, y_data, z_data = all_data[:, x_idx], all_data[:, y_idx], all_data[:, z_idx]
    # print('z_data shape', z_data.shape)
    return {'x': x_data.T.flatten(), 'y': y_data.T.flatten(), 'z': z_data.T.flatten()}

  def optimize(self):
    for idx in self.likelihoods:
      likelihood = self.likelihoods[idx]
      train_data = self.train_datas[idx]
      model = self.models[idx]
      optim = self.optimizers[idx]
      mll = self.mlls[idx]
      model.train()
      likelihood.train()
      model.set_train_data((self.train_time, self.train_task))
      # model train and optimize phase
      # Find optimal model hyperparameters
      with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=True), \
        gpytorch.settings.max_cg_iterations(100), \
        gpytorch.settings.max_preconditioner_size(80):
        for i in range(self.max_iter):
          optim.zero_grad()
          output = model(self.train_time, self.train_task)
          loss = -mll(output, train_data)
          loss.backward()
          optim.step()
          if (i + 1) % 10 == 0:
            print('Iter %d/%d - Loss: %.6f' % (i + 1, self.max_iter, loss.item()))

  def predict(self, pred_data):
    # todo: judge the point label
    pass

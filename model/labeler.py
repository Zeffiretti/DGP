###########################################################################
###                    Optic Data Auto-label Program                    ###
###                          ZEFFIRETTI, HIESH                          ###
###                   Beijing Institute of Technology                   ###
###                zeffiretti@bit.edu.cn, hiesh@mail.com                ###
###########################################################################

import torch
import gpytorch
from .modgp import MultioutputDGPModel
import warnings
from gpytorch.mlls import SumMarginalLogLikelihood

NUM_NAN = 1000


class Label:
  def __init__(self, init_time, init_data, device=torch.device('cpu'), max_iter=100):
    self.device = device
    assert init_data.shape[1] % 3 == 0, "init data shape error: some points may not contain 3-axis data"
    self.num_output = init_data.shape[1] // 3
    print('output num', self.num_output)
    self.train_datas = self.wrap_data(init_data, flatten=True)
    self.train_time, self.train_task = self.wrap_time(init_time, self.num_output)
    self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device),
                        gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device),
                        gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_output).to(self.device)]
    # next step: replace dict models with ModelList
    # self.models = [MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[0],
    #                                    self.likelihoods[0], self.num_output).to(self.device),
    #                MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[1],
    #                                    self.likelihoods[1], self.num_output).to(self.device),
    #                MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[2],
    #                                    self.likelihoods[2], self.num_output).to(self.device)]
    modelx = MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[0],
                                 self.likelihoods[0], self.num_output).to(self.device)
    modely = MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[1],
                                 self.likelihoods[1], self.num_output).to(self.device)
    modelz = MultioutputDGPModel((self.train_time, self.train_task), self.train_datas[2],
                                 self.likelihoods[2], self.num_output).to(self.device)
    # modelx.set_train_data((self.train_time, self.train_task))
    modelz.set_train_data((self.train_time, self.train_task), self.train_datas[2])
    self.waist_models = gpytorch.models.IndependentModelList(modelx, modely, modelz).to(self.device)
    self.waist_likelihoods = gpytorch.likelihoods.LikelihoodList(modelx.likelihood,
                                                                 modely.likelihood,
                                                                 modelz.likelihood).to(self.device)
    self.waist_optimizer = torch.optim.Adam([{'params': self.waist_models.parameters()}, ], lr=0.1)
    self.waist_mll = SumMarginalLogLikelihood(self.waist_likelihoods, self.waist_models).to(self.device)
    # self.optimizers = [torch.optim.Adam([{'params': self.models[0].parameters()}, ], lr=0.3),
    #                    torch.optim.Adam([{'params': self.models[1].parameters()}, ], lr=0.3),
    #                    torch.optim.Adam([{'params': self.models[2].parameters()}, ], lr=0.3),
    #                    ]  # Includes GaussianLikelihood parameters
    # self.mlls = {0: gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[0], self.models[0]).to(device),
    #              1: gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[1], self.models[1]).to(device),
    #              2: gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[2], self.models[2]).to(device), }
    self.err_bound = 0.2
    # for axis in self.models:
    #   print(axis, ':', self.models[axis])
    # print(next(self.models[axis].parameters()).device)
    self.max_iter = max_iter
    self.predications = None
    self.uppers = None
    self.lowers = None

    # def init_model(self, train_x, train_y, likelihood, num_outputs):
    #   return MultioutputDGPModel(train_x, train_y, likelihood, num_outputs)

  def set_train_data(self, train_time, train_data=None):
    update_train_x = self.wrap_time(train_time, self.num_output)
    if train_data is None:
      for model in self.waist_models.models:
        model.set_train_data(update_train_x)
    else:
      update_train_y = self.wrap_data(train_data, flatten=True)
      for yi, model in zip(update_train_y, self.waist_models.models):
        model.set_train_data(update_train_x, yi)

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

  def wrap_data(self, all_data, flatten=True):
    """
    split data containing 3-axis features into each channel, with flatten shape
    :param all_data:
    :type all_data: torch.tensor, rowCnt: N, colCnt: num_output x 3
    :param flatten: for train data, faltten should set True(default)
    :return: if not flatten (test), return 3x14 tensor
    """
    # print('all data shape', all_data.shape)
    colCnt = all_data.shape[1]
    assert colCnt % 3 == 0, "data shape error: some points may not contain 3-axis data"
    rowCnt = all_data.shape[0]
    x_idx = list(filter(lambda x: x % 3 == 0, range(colCnt)))
    y_idx = list(filter(lambda y: y % 3 == 1, range(colCnt)))
    z_idx = list(filter(lambda z: z % 3 == 2, range(colCnt)))
    x_data, y_data, z_data = all_data[:, x_idx], all_data[:, y_idx], all_data[:, z_idx]
    # print('z_data shape', z_data.shape)
    # return {0: x_data.T.flatten(), 1: y_data.T.flatten(), 2: z_data.T.flatten()} if flatten \
    #   else torch.cat([x_data, y_data, z_data], dim=0)
    return [x_data.T.flatten(), y_data.T.flatten(), z_data.T.flatten()] if flatten \
      else torch.cat([x_data, y_data, z_data], dim=0)

  def optimize(self, max_iter=None):
    if not max_iter is None:
      self.max_iter = max_iter
    self.waist_models.train()
    self.waist_likelihoods.train()
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=True), \
      gpytorch.settings.max_cg_iterations(100), \
      gpytorch.settings.max_preconditioner_size(80):
      for i in range(self.max_iter):
        self.waist_optimizer.zero_grad()
        output = self.waist_models(*self.waist_models.train_inputs)
        loss = -self.waist_mll(output, self.waist_models.train_targets)
        loss.backward()
        if (i + 1) % 10 == 0:
          print('Iter %d/%d - Loss: %.3f' % (i + 1, self.max_iter, loss.item()))
        self.waist_optimizer.step()
    # for idx in self.likelihoods:
    #   likelihood = self.likelihoods[idx]
    #   train_data = self.train_datas[idx]
    #   model = self.models[idx]
    #   optim = self.optimizers[idx]
    #   mll = self.mlls[idx]
    #   model.train()
    #   likelihood.train()
    #   model.set_train_data((self.train_time, self.train_task))
    #   # model train and optimize phase
    #   # Find optimal model hyperparameters
    #   with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=True), \
    #     gpytorch.settings.max_cg_iterations(100), \
    #     gpytorch.settings.max_preconditioner_size(80):
    #     for i in range(self.max_iter):
    #       optim.zero_grad()
    #       output = model(self.train_time, self.train_task)
    #       loss = -mll(output, train_data)
    #       loss.backward()
    #       optim.step()
    #       if (i + 1) % 10 == 0:
    #         print('Iter %d/%d - Loss: %.6f' % (i + 1, self.max_iter, loss.item()))

  def predict(self, pred_time, pred_data) -> torch.tensor:
    # todo: judge the point label use waist_model
    self.test_time, self.test_task = self.wrap_time(pred_time, self.num_output)
    test_datas = self.wrap_data(pred_data, flatten=False)
    num_test = test_datas.shape[1]
    # print('test points number is ', num_test)

    if num_test < self.num_output:
      warnings.warn('test points are less than train points, which means prediction is to take effect')
    if num_test > self.num_output:
      warnings.warn('test points are more than train points, which means extra points are to be removed')

    # initialize label results (matrix): channels(3) \times num_output \times num_test with value zero
    labels = torch.zeros((3, self.num_output, num_test), dtype=torch.int).to(self.device)
    # indexes = {0: 0, 1: 1, 2: 2}

    self.waist_models.eval()
    self.waist_models.eval()
    with torch.no_grad(), \
      gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=True), \
      gpytorch.settings.max_cg_iterations(100), \
      gpytorch.settings.max_preconditioner_size(80):
      predictions = self.waist_likelihoods(*self.waist_models([self.test_time, self.test_task],
                                                              [self.test_time, self.test_task],
                                                              [self.test_time, self.test_task]))
      # print(predictions)
      l_pred = None
      l_upp = None
      l_low = None
      # make identification on each point and each channel
      for i in range(self.num_output):  # i stands for point index
        for channel, prediction in enumerate(predictions):  # idx stands for channel
          # for channel, prediction in enumerate(predictions):  # idx stands for channel
          # print(channel, ':', prediction)  # each predication contains range of  num_outputs
          lower, upper = prediction.confidence_region()
          mean = prediction.mean.detach()
          # print(mean.shape)
          # confidence interval clamp
          # print('range ', ':', lower, upper)
          # for i in range(self.num_output):  # i stands for point index
          if upper[i] - lower[i] < 2 * self.err_bound:
            upper[i] = mean[i] + self.err_bound
            lower[i] = mean[i] - self.err_bound
          # print('upper type:', type(upper))
          # l_pred=torch.cat((l_pred))
          # pred_i = torch.tensor([mean[i]])
          l_pred = torch.tensor([[mean[i]]]).to(self.device) if l_pred is None \
            else torch.cat((l_pred, torch.tensor([[mean[i]]]).to(self.device)), dim=1)
          l_low = lower[i].view(1, -1) if l_low is None else torch.cat((l_low, lower[i].view(1, -1)), dim=1)
          l_upp = upper[i].view(1, -1) if l_upp is None else torch.cat((l_upp, upper[i].view(1, -1)), dim=1)
          for j in range(num_test):
            if lower[i] < test_datas[channel, j] < upper[i]:
              # print(lower[i], test_data[j], upper[i])
              labels[channel, i, j] = 1
    labels_and = torch.logical_and(torch.logical_and(labels[0, :, :], labels[1, :, :]), labels[2, :, :])
    result_matrix = labels_and.double()
    # wrap predications
    # [lower1 mean1 upper1,...]
    self.predications = l_pred if self.predications is None else torch.cat((self.predications, l_pred), dim=0)
    self.lowers = l_low if self.lowers is None else torch.cat((self.lowers, l_low), dim=0)
    self.uppers = l_upp if self.uppers is None else torch.cat((self.uppers, l_upp), dim=0)
    # print('result label is\n', labels)
    # labels_and should be permutation matrix
    # print('result rank is', torch.linalg.matrix_rank(result_matrix))
    # col_sum, row_sum = torch.sum(result_matrix, 0), torch.sum(result_matrix, 1)
    # # print('sum at dim 0', col_sum)
    # # print('sum at dim 1', row_sum)
    # # print('non zero ar column sum', torch.nonzero(col_sum))
    # # print('non zero ar row sum', torch.nonzero(row_sum))
    # lost_points = torch.where(col_sum == 0)
    # conflict_points = torch.where(col_sum > 1)
    # lost_labels = torch.where(row_sum == 0)
    # conflict_labels = torch.where(row_sum > 1)
    # print('summery', lost_labels, lost_points, conflict_labels, conflict_points)
    # print('shape', lost_labels[0].shape, lost_points[0].shape, conflict_labels[0].shape, conflict_points[0].shape)
    # labeled_data = torch.randn((self.num_output, 1, 3), dtype=torch.float64)
    # if lost_labels[0].shape[0] == 0 and conflict_labels[0].shape[0] == 0 and conflict_points[0].shape[0] == 0:
    #   # All labels are assigned correctly
    #   print('result after and operation is\n', result_matrix)
    #   print('good!')
    #   for channel in range(3):
    #     labeled_data[:, :, channel] = torch.matmul(result_matrix,
    #                                                torch.nan_to_num(test_datas[channel, :].view(-1, 1), nan=NUM_NAN))
    #     # print(a.shape)
    #   # print('unlabeled:\n', test_datas[0, :])
    #   # print('labeled:\n', labeled_data)
    # else:
    #   # handle abnormal exceptions
    #   # exception 1: lost_labels==lost_points
    #   # exception 2: lost labels have no counter-points
    #   # exception 3: conflicts on one label, namely, one label has two or more points
    #   # exception 4: conflicts on one point, namely, one point is recognized as more than one label
    #   if lost_labels[0].shape[0] > 0:
    #     print('lost labels:', lost_labels[0])
    #     if lost_points[0].shape[0] > 0:
    #       print('and there are lost points:', lost_points)
    #   if conflict_labels[0].shape[0] > 0:
    #     pass
    #   pass

    # for index in lost_points[0]:
    #   print(index, test_datas[:, index])
    # return labeled_data.view(1, -1)
    return result_matrix, self.predications[-1, :]  # this matrix should be handled outside the class

  def get_predications(self):
    return self.predications

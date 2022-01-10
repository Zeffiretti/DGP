###########################################################################
###                    Optic Data Auto-label Program                    ###
###                          ZEFFIRETTI, HIESH                          ###
###                   Beijing Institute of Technology                   ###
###                zeffiretti@bit.edu.cn, hiesh@mail.com                ###
###########################################################################

import torch
import gpytorch
import numpy as np
from .labeler import Label
from .modgp import MultioutputDGPModel
from .interpolation import QuinticPolynomial
import multiprocessing
from multiprocessing import Process
import os
from scipy.io import savemat
import warnings
import timeit
import scipy
import matplotlib.pyplot as plt
from dg_data import GaussianData
from torch.utils.data import DataLoader


class GaussianAgent:
  """
  This class is to combine the results from 4 different Labelers.
  """

  def __init__(self, all_data: np.ndarray, gaussian_window=120, save_mat=None, normalize=False, start_idx=0):
    super(GaussianAgent, self).__init__()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    self.data_set = GaussianData(all_data * 30)
    self.data_loader = DataLoader(self.data_set, pin_memory=True)
    self.train_data = None  # update this varible in for loop,shape :[gaussian_window,1]
    self.gaussian_size = gaussian_window
    self.g_number = [0, 4 * 3, 7 * 3, 11 * 3, 14 * 3]
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.iter = 10

    self.lost_start_idx = torch.ones((14, 1)) * -1
    self.lost_end_idx = torch.ones((14, 1)) * -1

    start_time = timeit.default_timer()
    self.run()  # runrunrun
    end_time = timeit.default_timer()
    print("consume time:", (end_time - start_time) / 60, 'min')

  def reset_model(self):
    self.split_data(self.train_data[:, 1:])  # rearrange all data into four groups
    labels_master = []
    for g in self.groups:
      labels_master.append(Label(self.train_data[:, 0], g, self.device))
    self.masters = labels_master

  def run(self):
    gap = 0
    for iter, data in enumerate(self.data_loader):
      gap += 1
      self.train_data = data
      if iter % 2000 == 0:
        self.reset_model()
        self.train_model(max_iter=150)
        gap = 0
      elif self.retrain:
        self.train_model(max_iter=100)
        gap = 0
      elif gap == 200:
        self.train_model(max_iter=self.iter)
        gap = 0
        self.iter = 10
      self.predict()
      print('\033[44m\033[1;30mround \033[1;31m {0:6d} \033[1;30m finished at \033[1;31m {1:3.6f} \033[1;30ms! \033[0m' \
            .format(iter, self.test_time.item() / 30))

    # todo: running process
    self.reset_model()
    self.train_model()
    self.predict()
    print('round 1 finished!')

    # self.fig, self.ax = plt.subplots()
    while self.tail_index < 29878:
      self.update_data()
      if self.tail_index % 2000 == 0:
        self.iter = 150
        self.reset_model()
        self.train_model(max_iter=self.iter)
        gap = 0
      if self.retrain:
        self.train_model(max_iter=100)
        gap = 0
        self.retrain = False
      else:
        if gap == 200:
          self.train_model(max_iter=self.iter)
          gap = 0
          self.iter = 10
      self.predict()
      print('\033[44m\033[1;30mround \033[1;31m {0:6d} \033[1;30m finished at \033[1;31m {1:3.6f} \033[1;30ms! \033[0m' \
            .format(self.tail_index, self.test_time.item() / 30))
    # print('checks is ', self.checks)
    res_mean_mat = None
    res_upper_mat = None
    res_lower_mat = None
    for master in self.masters:
      mean = master.predications
      upper = master.uppers
      lower = master.lowers
      res_mean_mat = mean if res_mean_mat is None else torch.cat((res_mean_mat, mean), dim=1)
      res_upper_mat = upper if res_upper_mat is None else torch.cat((res_upper_mat, upper), dim=1)
      res_lower_mat = lower if res_lower_mat is None else torch.cat((res_lower_mat, lower), dim=1)
    # print(res_mean_mat.shape)
    m_dic = {'time': self.valid_times.cpu().numpy() / 30,
             'mean': res_mean_mat.cpu().numpy() / 30,
             'upper': res_upper_mat.cpu().numpy() / 30,
             'lower': res_lower_mat.cpu().numpy() / 30}
    if self.save_mat is not None:
      savemat(self.save_mat, m_dic)

  def train_model(self, max_iter=150):
    # training phase
    model = 0
    for master in self.masters:
      master.optimize(max_iter=max_iter)
      if max_iter > 5:
        print(f'\033[44m\033[1;30mSub-model {model} trained.\033[0m')
        model += 1

  def predict(self):
    # self.tail_index += 1
    # self.header_index = self.tail_index + self.gaussian_window
    self.test_time = self.all_data[self.header_index, 0].detach().view(-1)
    self.test_pos = self.all_data[self.header_index, 1:].detach().view(1, -1)
    self.valid_times = self.test_time.view(-1, 1) if self.valid_times is None \
      else torch.cat((self.valid_times, self.test_time.view(-1, 1)), dim=0)

    self.permutation = None
    self.pred = None
    for master in self.masters:
      permu, mean = master.predict(self.test_time, self.test_pos)
      self.pred = mean if self.pred is None else torch.cat((self.pred, mean), 0)
      self.permutation = permu if self.permutation is None else torch.cat((self.permutation, permu), 0)
    # print('predict finished, and the result is\n')
    # print(self.permutation)
    # fit the matrix
    self.fit()
    if torch.all((self.permutation.matmul(self.permutation.T)).eq(torch.eye(14).to(self.device))):
      self.checks.append(1)
    else:
      self.checks.append(0)
      warnings.warn("predict failed at {0}".format(self.header_index))
      print(self.permutation)

  def split_data(self, data):
    """
    rearrange all data into four groups
    :param data: all data to be split
    :type data: torch.tensor, Nx42. column number should be 42 in order ...
    :return:
    """
    # the two asserts ensure input data shape
    assert data.shape[1] % 3 == 0, "input data does not contain all 3 channels for points, plz check it."
    point_number = data.shape[1] / 3
    assert point_number == 14, "input data number is not equal to 14, plz check it."
    time_number = data.shape[0]
    groups = []
    # [waist_group Rat 1, head_group Rat 1, waist_group Rat 2, head_group Rat 2]
    for i in range(4):
      groups.append(data[:, self.g_number[i]:self.g_number[i + 1]])
    self.groups = groups

  def decouple_data(self, data):
    """
    decouple all datas into 3 channels
    :param data: 1x42
    :return: 14x3
    """
    return data.view(14, 3)
    pass

  def update_data(self):
    self.tail_index += 1
    self.header_index = self.tail_index + self.gaussian_window
    update_time = self.all_data[self.tail_index:self.header_index, 0]
    # print('train data is', self.all_data[self.header_index - 1, 10:13])
    self.split_data(self.all_data[self.tail_index:self.header_index, 1:])  # rearrange all data into four groups
    for master, g in zip(self.masters, self.groups):
      master.set_train_data(update_time, g)
    # plt.clf()
    # plt.plot(update_time.cpu() / 30, self.all_data[self.tail_index:self.header_index, 1:].cpu())
    # plt.show()

  def fit(self):
    """
    fit the permutation matrix correctly
    :return:
    """
    # the not-permuted matrix are to be handled
    self.handled_data = self.test_pos.detach().view(14, 3)
    self.interpolation_idxes = []
    if not torch.all((self.permutation.matmul(self.permutation.T)).eq(torch.eye(14).to(self.device))):
      print('before fitting,', self.test_pos)
      self.predication_data = self.pred.detach().view(14, 3)
      self.last_data = self.all_data[self.header_index - 1, 1:].detach().view(14, 3)
      assert not torch.isnan(torch.sum(self.predication_data)), 'predication contains nan number at {0}'.format(self.header_index)
      self.fit_conflict_labels()
      print('after step 1,\n', self.permutation)
      self.fit_conflict_points()
      print('after step 2,\n', self.permutation)
      # todo: before fit lost points, recovery points are to be checked
      for i, lost_start in enumerate(self.lost_start_idx):
        if lost_start >= 0 and torch.sum(self.permutation[:, i]) == 1:
          self.lost_end_idx[i] = self.header_index
          self.interpolation_idxes.append(i)
      self.fit_lost()
      # no need to explicitly change the items
      # print('before,', self.test_pos)
      # self.test_pos = self.handled_data.view(1, -1)
      # print('after,', self.test_pos)
    else:
      self.iter = 10
    # print('handled data is', self.handled_data)
    assert not torch.isnan(torch.sum(self.handled_data)), 'handled_data contains nan number at {0}'.format(self.header_index)
    self.all_data[self.header_index, 1:] = self.handled_data.detach().view(1, -1)
    # print('after set,', self.all_data[self.header_index, 1:])
    for idx in self.interpolation_idxes:
      self.backward_interpolation(idx)

  def fit_conflict_labels(self):
    """

    :return:
    """
    col_sum, row_sum = torch.sum(self.permutation, 0), torch.sum(self.permutation, 1)
    # step 1: fix all conflict labels
    # howto: evaluate distance from the conflict points to each label, and pair the nearest
    conflict_labels = torch.where(row_sum > 1)[0]  # 0th of the tuple contains one tensor
    print('before step 1,\n', self.permutation)
    for conflict_label_idx in conflict_labels:
      print('labels conflict at ', conflict_label_idx)
      print('conflict details:\n', self.permutation[conflict_label_idx, :])
      nebulous_idxes = torch.where(self.permutation[conflict_label_idx, :].detach() == 1)[0]
      # print("nebulous_idx is \n", nebulous_idxes)
      row = conflict_label_idx
      distance = 1000
      col = 0
      for nebulous_idx in nebulous_idxes:
        self.permutation[row, nebulous_idx] = 0
        warnings.warn('nebulous labels!!!')
        # print('predication is', predication_data[row, :])  # the latest predications
        # print('handled data is', self.handled_data[nebulous_idx, :])
        cdist = torch.cdist(self.handled_data[nebulous_idx, :].view(1, -1), self.last_data[row, :].view(1, -1))
        if cdist < distance:
          col = nebulous_idx
          distance = cdist
        # print('distance is', distance)
        # point = self.handled_data[row, :]
      self.permutation[row, col] = 1.0

  def fit_conflict_points(self):
    """

    :return:
    """
    col_sum, row_sum = torch.sum(self.permutation, 0), torch.sum(self.permutation, 1)
    conflict_points = torch.where(col_sum > 1)[0]  # 0th of the tuple contains one tensor
    for conflict_point_idx in conflict_points:
      print('points conflict at ', conflict_point_idx)
      print('conflict details:\n', self.permutation[:, conflict_point_idx])
      nebulous_idxes = torch.where(self.permutation[:, conflict_point_idx].detach() == 1)[0]
      # print("nebulous_idx is \n", nebulous_idxes)
      col = conflict_point_idx
      distance = 1000
      row = 0
      for nebulous_idx in nebulous_idxes:
        self.permutation[nebulous_idx, col] = 0
        warnings.warn('nebulous points!!!')
        # print('predication is', predication_data[col, :])  # the latest predications
        # print('handled data is', self.handled_data[nebulous_idx, :])
        cdist = torch.cdist(self.handled_data[nebulous_idx, :].view(1, -1), self.last_data[col, :].view(1, -1))
        if cdist < distance:
          row = nebulous_idx
          distance = cdist
        # print('distance is', distance)
        # point = self.handled_data[row, :]
      self.permutation[row, col] = 1.0

  def fit_lost(self):
    """

    :return:
    """
    col_sum, row_sum = torch.sum(self.permutation, 0), torch.sum(self.permutation, 1)
    conflict_labels = torch.where(row_sum > 1)[0]  # 0th of the tuple contains one tensor
    assert conflict_labels.shape[0] == 0, "conflict still exists after step 1!"
    lost_points = torch.where(col_sum == 0)[0]  # 🙂
    lost_labels = torch.where(row_sum == 0)[0]  # 0th of the tuple contains one tensor
    # if not lost_labels.shape[0] == lost_points.shape[0]:
    #   print(self.permutation)
    assert lost_labels.shape[0] == lost_points.shape[0], 'number of lost labels and points are not equal, ' \
                                                         'please handle it.'
    for idx, lost_label_idx in enumerate(lost_labels):
      # print('lost label index is', lost_label_idx, idx)
      row = lost_label_idx
      col = lost_points[idx]
      self.permutation[row, col] = 1
      print('row={0},col={1}'.format(row, col))
      print('last data is', self.last_data[col, :])
      print('predication is', self.predication_data[col, :])
      print('and data is ', self.handled_data[col, :])
      print('nan check is', torch.isnan(self.handled_data[col, :]))
      distance = torch.cdist(self.handled_data[col, :].view(1, -1), self.last_data[col, :].view(1, -1))
      # todo: replace fit param dilatometric with posture check
      if torch.sum(torch.isnan(self.handled_data[col, :]) == True) == 3 \
        or distance > self.loose_fit[col, 0]:
        # if the actual observed data is NaN or too large on cdist, replace it with predications
        # print('good, fit it!😋')
        self.handled_data[col, :] = self.predication_data[col, :]
        self.loose_fit[col, 0] *= self.exp
        # warnings.warn("replace raw data with predication at {0}".format(self.header_index))
        if self.lost_start_idx[col] == -1:
          self.lost_start_idx[col] = self.header_index
      else:
        # catch the lost value
        if self.lost_start_idx[col] > 0:
          self.lost_end_idx[col] = self.header_index
          self.interpolation_idxes.append(col)
        self.loose_fit[col, 0] = 2
        # warnings.warn("reset loose fit to 1 at {0}".format(self.header_index))
      print('\033[43mDistance is', distance, 'and loose fit is', self.loose_fit[col, 0], '\033[0m')
      pred_distance = torch.cdist(self.handled_data[col, :].view(1, -1), self.predication_data[col, :].view(1, -1))
      print('\033[41mPred_distance is', pred_distance, '\033[0m')
      self.iter = 10 if self.iter == 10 and (torch.isnan(pred_distance) or pred_distance < 0.8) else 30

  def backward_interpolation(self, index):
    """
    to interpolate the lost data backwardly, then how to ....
    :return:
    """
    # assertion
    assert self.lost_start_idx[index] >= 0, f"lost_start_idx is negative at {index}"
    assert self.lost_end_idx[index] >= 0, f"lost_end_idx is negative at {index}"
    assert self.lost_end_idx[index] > self.lost_start_idx[index], f"lost end index is not greater than lost start index at {index}"
    # rethink: is it necessary to consider the differential range? now we prefer not
    print('data at {0} should be interpolated.'.format(index))
    print('start at {0}, end at {1}'.format(self.lost_start_idx[index], self.lost_end_idx[index]))
    # print('all start is ', self.lost_start_idx)
    # print('all end is ', self.lost_end_idx)
    # todo next: consider how many points are used to inference the lost point
    # todo next: decide in which method to interpolate the series
    # todo next: overwrite the lost data
    # todo next: check geometry satisfaction
    s_idx, e_idx = self.lost_start_idx[index].int().item(), self.lost_end_idx[index].int().item()
    # print('before lost data:', self.all_data[s_idx - 5:e_idx + 2, :])
    lost = self.all_data[s_idx - 5:e_idx + 2, :].cpu().numpy()
    lost_data = {'lost': lost}
    savemat(self.save_mat, lost_data)
    point_index_x = index * 3 + 1

    start_x, start_y, start_z = self.all_data[s_idx, point_index_x:point_index_x + 3]
    final_x, final_y, final_z = self.all_data[e_idx, point_index_x:point_index_x + 3]
    start_time, final_time = self.all_data[[s_idx, e_idx], 0]
    print(f"interpolate start {start_time, start_x, start_y, start_z}")
    quint_x = QuinticPolynomial(0, start_x, 0, 0, final_time - start_time, final_x, 0, 0, device=self.device)
    quint_y = QuinticPolynomial(0, start_y, 0, 0, final_time - start_time, final_y, 0, 0, device=self.device)
    quint_z = QuinticPolynomial(0, start_z, 0, 0, final_time - start_time, final_z, 0, 0, device=self.device)
    print(f"interpolate final {final_time, final_x, final_y, final_z}")

    for idx in range(s_idx, e_idx):
      t = self.all_data[idx, 0] - start_time
      self.all_data[idx, point_index_x] = torch.tensor([quint_x.get_position(t)]).to(self.device)
      self.all_data[idx, point_index_x + 1] = torch.tensor([quint_y.get_position(t)]).to(self.device)
      self.all_data[idx, point_index_x + 2] = torch.tensor([quint_z.get_position(t)]).to(self.device)
      print(f'interpolate at {t}:', self.all_data[idx, point_index_x:point_index_x + 3])
    print('backward interpolation finished!')
    self.retrain = True

    # after interpolation
    self.lost_start_idx[index] = -1
    self.lost_end_idx[index] = -1

    # raise Exception('aha🤪 stop!')

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
import multiprocessing
from multiprocessing import Process
import os
from scipy.io import savemat
import warnings


class DynamicAgent:
  """
  This class is to combine the results from 4 different Labelers.
  """

  def __init__(self, all_data: type(torch.tensor([])), gaussian_window=120, save_mat=None):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    self.tail_index = 0
    self.gaussian_window = gaussian_window
    self.header_index = self.tail_index + self.gaussian_window
    self.checks = []
    self.save_mat = save_mat
    self.loose_fit = 1
    self.exp = 1.0 + 1e-3
    # self.all_data = torch.nan_to_num(all_data, nan=1000)
    self.all_data = all_data
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.valid_times = None
    if not isinstance(self.all_data, type(torch.tensor([]))):
      raise TypeError("input data variables not of type torch.tensor")
    print('input date shape:', self.all_data.shape)
    self.all_data.to(self.device)
    print('before notmalize: \n', self.all_data[:10, :4])
    self.all_data = torch.nan_to_num(self.all_data, nan=0)
    self.all_data[:, 1:] = torch.nn.functional.normalize(self.all_data[:, 1:], dim=0) * 300
    print('after notmalize: \n', self.all_data[:10, :4])
    # todo: 4 labelers are to be created
    self.g_number = [0, 4 * 3, 7 * 3, 11 * 3, 14 * 3]
    self.split_data(self.all_data[self.tail_index:self.header_index, 1:])  # rearrange all data into four groups
    labels_master = []
    for g in self.groups:
      labels_master.append(Label(self.all_data[self.tail_index:self.header_index, 0], g, self.device))
    self.masters = labels_master
    self.run()  # runrunrun

  def run(self):
    # todo: running process
    self.train_model()
    self.predict()
    print('round 1 finished!')
    while self.tail_index < 15000:
      self.update_data()
      self.train_model(max_iter=5)
      self.predict()
      warnings.warn('round {0} finished at {1}!'.format(self.tail_index, self.test_time))
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
    for master in self.masters:
      master.optimize(max_iter=max_iter)

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
    print('train data is', self.all_data[self.header_index - 1, 10:13])
    self.split_data(self.all_data[self.tail_index:self.header_index, 1:])  # rearrange all data into four groups
    for master, g in zip(self.masters, self.groups):
      master.set_train_data(update_time, g)

  def fit(self):
    """
    fit the permutation matrix correctly
    :return:
    """
    # the not-permuted matrix are to be handled
    if not torch.all((self.permutation.matmul(self.permutation.T)).eq(torch.eye(14).to(self.device))):
      print('before pos,', self.test_pos)
      handled_data = self.test_pos.detach().view(14, 3)
      predication_data = self.pred.detach().view(14, 3)
      col_sum, row_sum = torch.sum(self.permutation, 0), torch.sum(self.permutation, 1)
      # step 1: process all conflict labels
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
          # print('handled data is', handled_data[nebulous_idx, :])
          cdist = torch.cdist(handled_data[nebulous_idx, :].view(1, -1), predication_data[row, :].view(1, -1))
          if cdist < distance:
            col = nebulous_idx
            distance = cdist
          # print('distance is', distance)
          point = handled_data[row, :]
        self.permutation[row, col] = 1.0
      print('after step 1,\n', self.permutation)

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
          # print('handled data is', handled_data[nebulous_idx, :])
          cdist = torch.cdist(handled_data[nebulous_idx, :].view(1, -1), predication_data[col, :].view(1, -1))
          if cdist < distance:
            row = nebulous_idx
            distance = cdist
          # print('distance is', distance)
          # point = handled_data[row, :]
        self.permutation[row, col] = 1.0
      print('after step 2,\n', self.permutation)

      col_sum, row_sum = torch.sum(self.permutation, 0), torch.sum(self.permutation, 1)
      conflict_labels = torch.where(row_sum > 1)[0]  # 0th of the tuple contains one tensor
      assert conflict_labels.shape[0] == 0, "conflict still exists after step 1!"
      lost_points = torch.where(col_sum == 0)[0]  # 🙂
      lost_labels = torch.where(row_sum == 0)[0]  # 0th of the tuple contains one tensor
      if not lost_labels.shape[0] == lost_points.shape[0]:
        print(self.permutation)
      assert lost_labels.shape[0] == lost_points.shape[0], 'number of lost labels and points are not equal, ' \
                                                           'please handle it.'
      for idx, lost_label_idx in enumerate(lost_labels):
        # print('lost label index is', lost_label_idx, idx)
        row = lost_label_idx
        col = lost_points[idx]
        self.permutation[row, col] = 1
        print('col={0},row={1}'.format(row, col))
        print('predication is', predication_data[row, :])
        print('and data is ', handled_data[row, :])
        print('nan check is', torch.isnan(handled_data[row, :]))
        distance = torch.cdist(handled_data[row, :].view(1, -1), predication_data[row, :].view(1, -1))
        if torch.sum(torch.isnan(handled_data[row, :]) == True) == 3 \
          or distance > self.loose_fit:
          # if the actual observed data is NaN or too large on cdist, replace it with predications
          # print('good, fit it!😋')
          handled_data[row, :] = predication_data[row, :]
          self.loose_fit *= self.exp
          warnings.warn("replace raw data with predication at {0}".format(self.header_index))
        else:
          self.loose_fit = 1
          warnings.warn("reset loose fit to 1 at {0}".format(self.header_index))
        print('distance is', distance, 'and loose fit is', self.loose_fit)
      # no need to explicitly change the items
      # print('before,', self.test_pos)
      # self.test_pos = handled_data.view(1, -1)
      print('after,', self.test_pos)
      self.all_data[self.header_index, 1:] = self.test_pos.detach()
      print('after set,', self.all_data[self.header_index, 1:])

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


class DynamicAgent:
  """
  This class is to combine the results from 4 different Labelers.
  """

  def __init__(self, all_data: type(torch.tensor([])), gaussian_window=120):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    self.tail_index = 0
    self.gaussian_window = gaussian_window
    self.header_index = 0 + self.gaussian_window
    self.checks = []
    self.all_data = torch.nan_to_num(all_data, nan=1000)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(self.all_data, type(torch.tensor([]))):
      raise TypeError("input data variables not of type torch.tensor")
    print('input date shape:', self.all_data.shape)
    self.all_data.to(self.device)
    # todo: 4 labelers are to be created
    self.g_number = [0, 4 * 3, 7 * 3, 11 * 3, 14 * 3]
    self.split_data(self.all_data[self.tail_index:self.header_index, 1:])  # rearrange all data into four groups
    labels_master = []
    for g in self.groups:
      labels_master.append(Label(self.all_data[self.tail_index:self.header_index, 0], g, self.device))
    self.masters = labels_master
    self.run()

  def run(self):
    # todo: running process
    self.train_model()
    self.predict()
    print('round 1 finished!')
    while self.tail_index < 20000:
      print('round {0} finished!'.format(self.tail_index))
      self.update_data()
      self.train_model(max_iter=5)
      self.predict()
    print('checks is ', self.checks)

  def train_model(self, max_iter=150):
    # training phase
    for master in self.masters:
      master.optimize(max_iter=max_iter)

  def predict(self):
    # self.tail_index += 1
    # self.header_index = self.tail_index + self.gaussian_window
    test_time = self.all_data[self.header_index, 0].view(-1)
    test_pos = self.all_data[self.header_index, 1:].view(1, -1)

    self.permutation = None
    for master in self.masters:
      permu = master.predict(test_time, test_pos)
      self.permutation = permu if self.permutation is None else torch.cat((self.permutation, permu), 0)
    print('predict finished, and the result is\n')
    print(self.permutation)
    if torch.all((self.permutation.matmul(self.permutation.T)).eq(torch.eye(14).to(self.device))):
      self.checks.append(1)
    else:
      raise "predict failed at {0}".format(self.header_index)
      self.checks.append(0)

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

  def update_data(self):
    self.tail_index += 1
    self.header_index = self.tail_index + self.gaussian_window
    update_time = self.all_data[self.tail_index:self.header_index, 0]
    self.split_data(self.all_data[self.tail_index:self.header_index, 1:])  # rearrange all data into four groups
    for master, g in zip(self.masters, self.groups):
      master.set_train_data(update_time, g)
      pass

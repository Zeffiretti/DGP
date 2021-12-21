import numpy as np
import torch


class QuinticPolynomial:
  def __init__(self, initial_time, initial_pos, initial_vel, initial_acc, final_time, final_pos, final_vel, final_acc, device=torch.device("cpu")):
    self.device=device
    self.position_coeff_ = torch.zeros((6, 1), dtype=torch.float).to(device)
    self.velocity_coeff_ = torch.zeros((6, 1), dtype=torch.float).to(device)
    self.acceleration_coeff_ = torch.zeros((6, 1), dtype=torch.float).to(device)
    self.time_varibles = torch.zeros((1, 6), dtype=torch.float).to(device)

    if final_time > initial_time:
      self.initial_time = initial_time
      self.initial_pos = initial_pos
      self.initial_vel = initial_vel
      self.initial_acc = initial_acc

      self.current_time = initial_time
      self.current_pos = initial_pos
      self.current_vel = initial_vel
      self.current_acc = initial_acc

      self.final_time = final_time
      self.final_pos = final_pos
      self.final_vel = final_vel
      self.final_acc = final_acc

      self.time_mat = torch.tensor(
        [[self.initial_time ** 5, self.initial_time ** 4, self.initial_time ** 3, self.initial_time ** 2, self.initial_time, 1.0],
         [5.0 * self.initial_time ** 4, 4.0 * self.initial_time ** 3, 3.0 * self.initial_time ** 2, 2.0 * self.initial_time, 1.0, 0.0],
         [20.0 * self.initial_time ** 3, 12.0 * self.initial_time ** 2, 6.0 * self.initial_time, 2.0, 0.0, 0.0],
         [self.final_time ** 5, self.final_time ** 4, self.final_time ** 3, self.final_time ** 2, self.final_time, 1.0],
         [5.0 * self.final_time ** 4, 4.0 * self.final_time ** 3, 3.0 * self.final_time ** 2, 2.0 * self.final_time, 1.0, 0.0],
         [20.0 * self.final_time ** 3, 12.0 * self.final_time ** 2, 6.0 * self.final_time, 2.0, 0.0, 0.0]
         ], dtype=torch.float).to(device)
      self.conditions_mat = torch.tensor([self.initial_pos, self.initial_vel, self.initial_acc, self.final_pos, self.final_vel, self.final_acc],
                                         dtype=torch.float).to(device)
      self.conditions_mat = self.conditions_mat.reshape((6, 1))
      self.position_coeff_ = torch.linalg.pinv(self.time_mat) @ self.conditions_mat
      self.velocity_coeff_ = torch.tensor([[0.0],
                                           [5.0 * self.position_coeff_[0, 0]],
                                           [4.0 * self.position_coeff_[1, 0]],
                                           [3.0 * self.position_coeff_[2, 0]],
                                           [2.0 * self.position_coeff_[3, 0]],
                                           [1.0 * self.position_coeff_[4, 0]],
                                           ]).to(device)
      self.acceleration_coeff_ = torch.tensor([[0.0],
                                               [0.0],
                                               [20.0 * self.position_coeff_[0, 0]],
                                               [12.0 * self.position_coeff_[1, 0]],
                                               [6.0 * self.position_coeff_[2, 0]],
                                               [2.0 * self.position_coeff_[3, 0]],
                                               ]).to(device)

  def get_position(self, t):
    if t >= self.final_time:
      self.current_time = self.final_time
      self.current_pos = self.final_pos
      self.current_vel = self.final_vel
      self.current_acc = self.final_acc
      return self.current_pos
    elif t <= self.initial_time:
      self.current_time = self.initial_time
      self.current_pos = self.initial_pos
      self.current_vel = self.initial_vel
      self.current_acc = self.initial_acc
      return self.current_pos
    else:
      self.current_time = t
      self.time_varibles = torch.tensor([t ** 5, t ** 4, t ** 3, t ** 2, t, 1.0], dtype=torch.float).to(self.device)
      self.current_pos = (self.time_varibles @ self.position_coeff_)[0]
      self.current_vel = (self.time_varibles @ self.velocity_coeff_)[0]
      self.current_acc = (self.time_varibles @ self.acceleration_coeff_)[0]
      return self.current_pos

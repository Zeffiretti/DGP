import numpy as np


class QuinticPolynomial:
  def __init__(self, initial_time, initial_pos, initial_vel, initial_acc, final_time, final_pos, final_vel, final_acc):
    self.position_coeff_ = np.zeros((6, 1), dtype=np.float)
    self.velocity_coeff_ = np.zeros((6, 1), dtype=np.float)
    self.acceleration_coeff_ = np.zeros((6, 1), dtype=np.float)
    self.time_varibles = np.zeros((1, 6), dtype=np.float)

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

      self.time_mat = np.array(
        [[self.initial_time ** 5, self.initial_time ** 4, self.initial_time ** 3, self.initial_time ** 2, self.initial_time, 1.0],
         [5.0 * self.initial_time ** 4, 4.0 * self.initial_time ** 3, 3.0 * self.initial_time ** 2, 2.0 * self.initial_time, 1.0, 0.0],
         [20.0 * self.initial_time ** 3, 12.0 * self.initial_time ** 2, 6.0 * self.initial_time, 2.0, 0.0, 0.0],
         [self.final_time ** 5, self.final_time ** 4, self.final_time ** 3, self.final_time ** 2, self.final_time, 1.0],
         [5.0 * self.final_time ** 4, 4.0 * self.final_time ** 3, 3.0 * self.final_time ** 2, 2.0 * self.final_time, 1.0, 0.0],
         [20.0 * self.final_time ** 3, 12.0 * self.final_time ** 2, 6.0 * self.final_time, 2.0, 0.0, 0.0]
         ], dtype=np.float)
      self.conditions_mat = np.array([self.initial_pos, self.initial_vel, self.initial_acc, self.final_pos, self.final_vel, self.final_acc], dtype=np.float)
      self.conditions_mat = self.conditions_mat.reshape((6, 1))
      self.position_coeff_ = np.linalg.pinv(self.time_mat) @ self.conditions_mat
      self.velocity_coeff_ = np.array([[0.0],
                                       [5.0 * self.position_coeff_[0, 0]],
                                       [4.0 * self.position_coeff_[1, 0]],
                                       [3.0 * self.position_coeff_[2, 0]],
                                       [2.0 * self.position_coeff_[3, 0]],
                                       [1.0 * self.position_coeff_[4, 0]],
                                       ])
      self.acceleration_coeff_ = np.array([[0.0],
                                           [0.0],
                                           [20.0 * self.position_coeff_[0, 0]],
                                           [12.0 * self.position_coeff_[1, 0]],
                                           [6.0 * self.position_coeff_[2, 0]],
                                           [2.0 * self.position_coeff_[3, 0]],
                                           ])

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
      self.time_varibles = np.array([t ** 5, t ** 4, t ** 3, t ** 2, t, 1.0], dtype=np.float)
      self.current_pos = (self.time_varibles @ self.position_coeff_)[0]
      self.current_vel = (self.time_varibles @ self.velocity_coeff_)[0]
      self.current_acc = (self.time_varibles @ self.acceleration_coeff_)[0]
      return self.current_pos

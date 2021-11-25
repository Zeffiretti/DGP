import torch

import RatInteractionDataLoader as ridl
import time
from model import Label


def main():
  data_dir = 'dataset/'

  label_prefix = 'smuro-interaction-'
  label_data = '1028'
  label_time = '-0834pm'
  label_suffix = '-short.csv'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print('device is', device)

  file_path = data_dir + label_data + '/' + label_prefix + label_data + label_time + label_suffix

  all_data = ridl.load_int_data(file_path, mformat_header=False).to_numpy()
  tensor_data = torch.tensor(all_data).to(device)
  # print(all_data[:3, :2])

  print(tensor_data.shape)

  t_dura_1s = 240  # 240 frames for 1s
  start_time = 0
  end_time = 10
  sample = 1
  gaussian_window = 24 * 5

  data_time_index = 0
  points_number = 3  # load 3 points to test
  data_time = tensor_data[:gaussian_window, 0]
  data_pos = tensor_data[:gaussian_window, 1:3 * points_number + 1]
  print('data device', data_time.device)
  print('data device', data_pos.device)

  labeler = Label(data_time, data_pos, device=device)

  labeler.optimize()


if __name__ == '__main__':
  main()

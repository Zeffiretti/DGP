import torch

import RatInteractionDataLoader as ridl
import time
from model import Label, dynamicagent


def main():
  data_dir = 'dataset/'

  label_prefix = 'smuro-interaction-'
  label_data = '1028'
  label_time = '-0834pm'
  # label_suffix = '-short.csv'
  label_suffix = '-restored.mat'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print('device is', device)

  file_path = data_dir + label_data + '/' + label_prefix + label_data + label_time + label_suffix

  # all_data = ridl.load_csv_data(file_path, mformat_header=False).to_numpy() * 60
  all_data = ridl.load_mat_file(file_path)['imputed_data']
  tensor_data = torch.tensor(all_data, dtype=torch.float64).to(device)
  # print(all_data[:3, :2])

  print(tensor_data.shape)

  t_dura_1s = 240  # 240 frames for 1s
  start_time = 0
  end_time = 10
  sample = 1
  gaussian_window = 24 * 5

  data_time_index = 0
  points_number = 4  # load 4 points to test
  data_time = tensor_data[:gaussian_window, 0]
  data_pos = tensor_data[:gaussian_window, 1:3 * points_number + 1]
  print('data device', data_time.device)
  print('data device', data_pos.device)

  labeler = Label(data_time, data_pos, device=device, max_iter=150)

  labeler.optimize()
  test_time = tensor_data[gaussian_window, 0].view(-1)  # time value should be Nx1 shape
  test_pos = tensor_data[gaussian_window, 1:].view(1, -1)
  aha = labeler.predict(test_time, test_pos)
  print('predication shape:', aha.shape)
  print('and its items:', aha)
  print('test pos is:', test_pos[0, :12])
  # aha shape should be [1,14*3]


def test():
  data_dir = 'dataset/'

  label_prefix = 'smuro-interaction-'
  label_data = '1028'
  label_time = '-0834pm'
  # label_suffix = '-short.csv'
  label_suffix = '-restored-matlab.mat'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print('device is', device)

  file_path = data_dir + label_data + '/' + label_prefix + label_data + label_time + label_suffix

  # all_data = ridl.load_csv_data(file_path, mformat_header=False).to_numpy() * 60
  all_data = ridl.load_mat_file(file_path)['imputed_data']
  tensor_data = torch.tensor(all_data, dtype=torch.float64).to(device)
  agent = dynamicagent.DynamicAgent(tensor_data, save_mat='result.mat', gaussian_window=120, normalize=False, start_idx=0)

  # data = ridl.load_mat_file('dataset/restored_data.mat')
  # print('data type', type(data))
  # print('data contents', data['imputed_data'][:3, :3])


if __name__ == '__main__':
  # main()
  test()

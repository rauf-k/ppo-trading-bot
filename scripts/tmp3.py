import os
from random import randrange
import numpy as np
import matplotlib.pyplot as plt


class DataProcessorIntraday:
    """
    how will this work?

    in production, you will append 1 timeslice at a time
    on another thread, you will get batch, if batch is none, that means you dont have enough
    datapoint yet

    in training, you will pass all the data from mysql or a text file
    not sure what format yet, but im curious to use a pandas dataframe
    in training, you will get a batch just like in a daily system

    """
    def __init__(
            self,
            in_training_mode: bool,
            observation_sequence_len=77,
            window_size_in_seconds=3.3,
            number_of_points_to_interpolate=32,
            train_trajectory_len=4000
    ):
        self.in_training_mode = in_training_mode

        self.observation_sequence_len = observation_sequence_len
        self.window_size_in_seconds = window_size_in_seconds
        self.number_of_points_to_interpolate = number_of_points_to_interpolate
        self.train_trajectory_len = train_trajectory_len
        self.train_observation_index = 0
        #
        self.key_tsls = None
        self.key_tsls__val_trajectorydata = None

    def _slice_lists_to_windows(self, ts_local_s_, price_, size_, window_size_in_seconds: float):
        key_tsls__val_windowdata = {}
        tsls = ts_local_s_.copy()
        price = price_.copy()
        size = size_.copy()
        tsls.reverse()
        price.reverse()
        size.reverse()
        window_tsls = []
        window_price = []
        window_size = []
        tsls_previous = tsls[randrange(300)] if self.in_training_mode else tsls[0]
        for tsls_val, price_val, size_val in zip(tsls, price, size):
            window_tsls.append(tsls_val)
            window_price.append(price_val)
            window_size.append(size_val)
            if tsls_previous - tsls_val >= window_size_in_seconds:
                window_tsls.reverse()
                window_price.reverse()
                window_size.reverse()
                key_tsls__val_windowdata[tsls_val] = {
                    'window_tsls': window_tsls,
                    'window_price': window_price,
                    'window_size': window_size,
                }
                window_tsls = []
                window_price = []
                window_size = []
                tsls_previous = tsls_val
        keys = list(key_tsls__val_windowdata.keys())
        keys.sort()
        key_tsls__val_windowdata_sorted = {}
        for k in keys:
            key_tsls__val_windowdata_sorted[k] = key_tsls__val_windowdata[k]
        return key_tsls__val_windowdata_sorted

    def _interpolate_single_window(self, window_tsls, window_data, number_of_points):
        xnew = np.linspace(min(window_tsls), max(window_tsls), num=number_of_points)
        ynew = np.interp(xnew, window_tsls, window_data)
        # return xnew, ynew
        return ynew

    def _process_all_windows(self, key_tsls__val_windowdata_sorted):
        key_tsls__val_windowsprocessed = {}
        for tsls_key in key_tsls__val_windowdata_sorted.keys():
            window_data = key_tsls__val_windowdata_sorted[tsls_key]

            window_tsls = window_data['window_tsls']
            window_price = window_data['window_price']
            window_size = window_data['window_size']

            value_n_points = float(len(window_tsls))
            value_price_std = np.std(window_price)
            value_size_std = np.std(window_size)
            # value_segment_duration = max(window_tsls) - min(window_tsls)  # this is wrong!

            window_price_i = self._interpolate_single_window(
                window_tsls, window_price, self.number_of_points_to_interpolate)
            window_size_i = self._interpolate_single_window(
                window_tsls, window_size, self.number_of_points_to_interpolate)

            """
            calculate standard deviation, mean, variance, ... 
            """

            key_tsls__val_windowsprocessed[tsls_key] = {
                'value_n_points': value_n_points,
                'value_price_std': value_price_std,
                'value_size_std': value_size_std,
                # 'value_segment_duration': value_segment_duration,

                'window_price_i': window_price_i,
                'window_size_i': window_size_i,
            }

        return key_tsls__val_windowsprocessed

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def train_load_from_txt_file(self, file_path):
        file1 = open(file_path, 'r')
        file_lines = file1.readlines()
        file1.close()
        price = []
        size = []
        timestamp_local = []
        for line in file_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            line_splitted = line.split(',')
            price.append(float(line_splitted[0]))
            size.append(float(line_splitted[1]))
            timestamp_local.append(float(line_splitted[3]))
        response = self.train_load_all_data(price, size, timestamp_local)
        return response

    def train_load_all_data(self, price_list_f, size_list_f, timestamp_ns_local_list_f):
        assert len(price_list_f) == len(size_list_f) == len(timestamp_ns_local_list_f)
        timestamp_sec_local_list_f = [float(i) / 1000000000.0 for i in timestamp_ns_local_list_f]
        key_tsls__val_windowdata_sorted = self._slice_lists_to_windows(
            timestamp_sec_local_list_f,
            price_list_f,
            size_list_f,
            self.window_size_in_seconds
        )
        if len(key_tsls__val_windowdata_sorted) <= self.train_trajectory_len + self.observation_sequence_len + 12:
            return False
        random_start = randrange(
            0,
            len(key_tsls__val_windowdata_sorted) - (self.train_trajectory_len + self.observation_sequence_len + 7)
        )
        random_end = random_start + (self.train_trajectory_len + self.observation_sequence_len + 3)
        key_tsls_full = list(key_tsls__val_windowdata_sorted.keys())
        key_tsls_trajectory = key_tsls_full[random_start: random_end]

        trajectory_raw = {kt: key_tsls__val_windowdata_sorted[kt] for kt in key_tsls_trajectory}
        self.key_tsls__val_trajectorydata = self._process_all_windows(trajectory_raw)
        self.key_tsls = list(self.key_tsls__val_trajectorydata.keys())

        return True

    def train_get_observation(self):
        """
        what to you get when you get a batch?
        you get a window of windows

        seq = [0, 1, 2, 3, 4, 5]
        window_size = 3

        for i in range(len(seq) - window_size + 1):
            print(seq[i: i + window_size])

        """
        observation_timestamps = self.key_tsls[
            self.train_observation_index: self.train_observation_index + self.observation_sequence_len]

        observation_data = {ots: self.key_tsls__val_trajectorydata[ots] for ots in observation_timestamps}

        self.train_observation_index = self.train_observation_index + 1

        return observation_data  # , self.key_tsls, observation_timestamps
    # ...............................................................................................................
    def prod_append_slice(self, price_value, size_value, timestamp_np_local_value):
        return None

    def prod_get_observation(self):
        return None


dpi = DataProcessorIntraday(in_training_mode=True, observation_sequence_len=77)

date_dir = 'data/train'
for file_name in os.listdir(date_dir):
    file_path = os.path.join(date_dir, file_name)
    # file_lines = get_file_lines(file_path)
    # price, size, ts_ib, ts_local = parse_file_lines(file_lines)
    # r = dpi.train_load_all_data(price, size, ts_local)
    r = dpi.train_load_from_txt_file(file_path)
    print(r, file_name)
    if r is False:
        continue

    while True:
        raw_observation_ = dpi.train_get_observation()
        print(len(raw_observation_))

    exit()
    """
    # print(observation_timestamps)
    for k in raw_observation:
        # print(k, raw_observation[k])
        for i in raw_observation[k]:
            # print(i, raw_observation[k][i])
        print('*' * 70)
    # exit()
    print('*' * 120)
    """



    # print(len(raw_observation), len(self_key_tsls), len(observation_timestamps))
    # print(self_key_tsls[0], observation_timestamps[0])
    # print(self_key_tsls[3999], observation_timestamps[3999])
    # for bv in b:
    #     print(bv)



    exit()


print('Done!')

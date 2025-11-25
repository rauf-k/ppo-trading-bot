import os
import json
import random
from random import randrange
from datetime import datetime
import numpy as np
from lib.ib_api import IbApi
from lib import const as CONST


class Market:
    def __init__(self, simulation_mode=True, real_money_mode=False):
        self.simulation_mode = simulation_mode
        self.real_money_mode = real_money_mode
        # ////////////////////////////////////////////////////////////////////////////////////////////////
        if not self.simulation_mode:
            self.ib_api = IbApi(
                host_ip_address=CONST.UTIL_TWS_HOST_IP,
                port_number=CONST.UTIL_TWS_PORT_PAPER_MONEY if not self.real_money_mode else CONST.UTIL_TWS_PORT_REAL_MONEY,
                client_id=CONST.UTIL_TWS_CLIENT_ID
            )
        else:
            self.ib_api = None
        # ////////////////////////////////////////////////////////////////////////////////////////////////
        if self.simulation_mode:
            self.real_money_mode = False
            self.market_sim_daily = MarketSimDaily(
                data_dir_train=CONST.UTIL_DIR_TRAIN,
                data_dir_val=CONST.UTIL_DIR_VAL,
                trajectory_len=CONST.STEPS_PER_EPOCH,
                number_of_days_in_window=CONST.OBSERVATION_WINDOW_LEN + 1
            )
            # ............................................................................
            self.symbols_train = [os.path.basename(i).rsplit('.', 1)[0].upper() for i in os.listdir(CONST.UTIL_DIR_TRAIN)]
            self.symbols_val = [os.path.basename(i).rsplit('.', 1)[0].upper() for i in os.listdir(CONST.UTIL_DIR_VAL)]
            for st in self.symbols_train:
                if st in self.symbols_val:
                    raise RuntimeError('Can not have same symbols in train and val: {}'.format(st))
            # ............................................................................
            self.current_symbol = None
        else:
            self.market_sim_daily = None
            self.symbols_train = None
            self.symbols_val = None
            self.current_symbol = None
        # ////////////////////////////////////////////////////////////////////////////////////////////////

    def get_market_data(self, symbol):
        if symbol not in self.symbols_train and symbol not in self.symbols_val:
            raise RuntimeError('Error: data for symbol {} dont exist'.format(symbol))

        if self.simulation_mode:  # you are using alpha vantage data
            if symbol == self.current_symbol:  # same symbol
                ib_data = self.market_sim_daily.today()  # what happens when you get to the end of a file?
                return ib_data
            else:  # new symbol
                success = self.market_sim_daily.set_new_symbol(symbol)
                if success:
                    ib_data = self.market_sim_daily.today()
                    self.current_symbol = symbol
                    return ib_data
                else:
                    return None
        else:  # get data from ib
            ib_data = self.ib_api.get_data_historical(symbol, number_of_months=2)
            # handle when symbol is undefined
            return ib_data


class MarketSimDaily:
    def __init__(
            self,
            data_dir_train,
            data_dir_val,
            trajectory_len,
            number_of_days_in_window,

            min_average_volume=200000,
            min_price_threshold=3,
            max_price_threshold=1200
    ):
        self.data_dir_train = data_dir_train
        self.files_list_train = os.listdir(self.data_dir_train)
        self.data_dir_val = data_dir_val
        self.files_list_val = os.listdir(self.data_dir_val)

        self.trajectory_len = trajectory_len
        self.number_of_days_in_window = number_of_days_in_window
        self.min_average_volume = min_average_volume
        self.min_price_threshold = min_price_threshold
        self.max_price_threshold = max_price_threshold

        self.market_data_dict = {}
        self.market_dates_list = []
        self.current_symbol = None
        self.observation_window_index = 0

    def _get_json_data(self, file_path):
        file1 = open(file_path)
        json_data = json.load(file1)
        file1.close()
        return json_data

    def _validate_file_market_data(self, json_data):
        closes = []
        volumes = []
        for date in json_data:
            closes.append(float(json_data[date]['4. close']))
            volumes.append(float(json_data[date]['5. volume']))
        if sum(volumes) / len(volumes) > self.min_average_volume:
            if min(closes) > self.min_price_threshold:
                if max(closes) < self.max_price_threshold:
                    return True
        return False

    def _convert_to_ib_format(self, json_data_av):
        key_dateunix__val_dateib = {}
        key_dateunix__val_dateav = {}
        for date_av in json_data_av:
            dt = datetime.strptime(date_av, '%Y-%m-%d')
            date_unix = dt.timestamp()
            key_dateunix__val_dateav[date_unix] = date_av
            date_ib = dt.strftime('%Y%m%d')
            key_dateunix__val_dateib[date_unix] = int(date_ib)

        timestamps = list(key_dateunix__val_dateav.keys())
        timestamps.sort()

        dates_ib_list = []
        data_ib_format = {}
        for date_unix in timestamps:
            data_av = json_data_av[key_dateunix__val_dateav[date_unix]]
            date_ib = key_dateunix__val_dateib[date_unix]
            data_ib_format[date_ib] = {
                'd': date_ib,
                'o': float(data_av['1. open']),
                'h': float(data_av['2. high']),
                'l': float(data_av['3. low']),
                'c': float(data_av['4. close']),
                'v': float(data_av['5. volume']),
                'b': 0.0
            }
            dates_ib_list.append(date_ib)

        return dates_ib_list, data_ib_format

    def reset(self, symbol=None):
        self.market_data_dict = {}
        self.market_dates_list = []
        self.current_symbol = None
        self.observation_window_index = 0

        if symbol is None:
            while True:
                random_file_name = random.choice(self.files_list_train)
                file_path = os.path.join(self.data_dir_train, random_file_name)
                json_data = self._get_json_data(file_path)['Time Series (Daily)']
                if len(json_data) < self.trajectory_len + self.number_of_days_in_window + 12:
                    continue
                key_stamp__val_date = {
                    datetime.strptime(date, '%Y-%m-%d').timestamp(): date for date in list(json_data.keys())}
                timestamps = list(key_stamp__val_date.keys())
                timestamps.sort()
                most_recent_timestamps = timestamps[: self.trajectory_len + self.number_of_days_in_window + 7]
                json_data = {
                    json_data[key_stamp__val_date[ts]]: key_stamp__val_date[ts] for ts in most_recent_timestamps}
                if self._validate_file_market_data(json_data):
                    self.current_symbol = random_file_name.rsplit('.', 1)[0]
                    self.market_dates_list, self.market_data_dict = self._convert_to_ib_format(json_data)
                    return True
        else:
            file_path = os.path.join(self.data_dir_val, '{}.json'.format(symbol))
            json_data = self._get_json_data(file_path)['Time Series (Daily)']
            if len(json_data) < self.number_of_days_in_window + 70:
                return False
            if self._validate_file_market_data(json_data):
                self.current_symbol = symbol
                self.market_dates_list, self.market_data_dict = self._convert_to_ib_format(json_data)
                return True
            return False

    def set_new_symbol(self, symbol):
        self.market_data_dict = {}
        self.market_dates_list = []
        self.current_symbol = None
        self.observation_window_index = 0

        # random_file_name = random.choice(self.files_list_train)

        file_name = '{}.json'.format(symbol)
        if file_name in self.files_list_train:  # it is a training file
            file_path = os.path.join(self.data_dir_train, file_name)
            json_data = self._get_json_data(file_path)['Time Series (Daily)']
            if len(json_data) < self.trajectory_len + self.number_of_days_in_window + 12:
                return False
            key_stamp__val_date = {
                datetime.strptime(date, '%Y-%m-%d').timestamp(): date for date in list(json_data.keys())}
            timestamps = list(key_stamp__val_date.keys())
            timestamps.sort()
            most_recent_timestamps = timestamps[: self.trajectory_len + self.number_of_days_in_window + 7]
            # json_data = {json_data[key_stamp__val_date[ts]]: key_stamp__val_date[ts] for ts in most_recent_timestamps}

            json_data = {key_stamp__val_date[ts]: json_data[key_stamp__val_date[ts]] for ts in most_recent_timestamps}

            if self._validate_file_market_data(json_data):
                self.current_symbol = file_name.rsplit('.', 1)[0]
                self.market_dates_list, self.market_data_dict = self._convert_to_ib_format(json_data)
                return True
            else:
                return False
        elif file_name in self.files_list_val:  # it is a validation file
            file_path = os.path.join(self.data_dir_val, file_name)
            json_data = self._get_json_data(file_path)['Time Series (Daily)']
            if len(json_data) < self.number_of_days_in_window + 70:
                return False
            if self._validate_file_market_data(json_data):
                self.current_symbol = symbol
                self.market_dates_list, self.market_data_dict = self._convert_to_ib_format(json_data)
                return True
            return False
        else:
            raise RuntimeError('Error: this {} file dont exist'.format(file_name))

    def today(self):
        dates_ib_for_window = self.market_dates_list[
            self.observation_window_index: self.observation_window_index + self.number_of_days_in_window]
        ib_data = {date_ib: self.market_data_dict[date_ib] for date_ib in dates_ib_for_window}
        self.observation_window_index = self.observation_window_index + 1

        return ib_data


class MarketSimIntraday_:
    def __init__(
            self,
            training_mode: bool,
            observation_sequence_len=77,
            window_size_in_seconds=3.3,
            number_of_points_to_interpolate=32,
            train_trajectory_len=4000
    ):
        self.in_training_mode = training_mode
        self.observation_sequence_len = observation_sequence_len
        self.window_size_in_seconds = window_size_in_seconds
        self.number_of_points_to_interpolate = number_of_points_to_interpolate
        self.train_trajectory_len = train_trajectory_len

        self.train_observation_index = 0
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
        self.train_observation_index = 0
        self.key_tsls = None
        self.key_tsls__val_trajectorydata = None
        #
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
        self.key_tsls__val_trajectorydata = {kt: key_tsls__val_windowdata_sorted[kt] for kt in key_tsls_trajectory}
        self.key_tsls = list(self.key_tsls__val_trajectorydata.keys())
        return True

    def train_get_observation(self):
        observation_timestamps = self.key_tsls[
            self.train_observation_index: self.train_observation_index + self.observation_sequence_len]
        observation_data = {ots: self.key_tsls__val_trajectorydata[ots] for ots in observation_timestamps}
        self.train_observation_index = self.train_observation_index + 1
        return observation_data
    # ...............................................................................................................
    def prod_append_slice(self, price_value, size_value, timestamp_np_local_value):
        return None

    def prod_get_observation(self):
        return None


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class MarketSimIntraday:
    def __init__(
            self,
            train_trajectory_len=4000,
            observation_sequence_len=1000,
            loop_time_fixed=3.0
    ):
        self.train_trajectory_len = train_trajectory_len
        self.observation_seq_len = observation_sequence_len
        self.loop_time_fixed = loop_time_fixed
        #
        self.key_stepindex__val_stepdata = {}

    def get_initial_timestamp_index(self, timestamps, start_time_range_s=77):
        index_final = 0
        timestamp_initial = timestamps[0]
        for i, t in enumerate(timestamps):
            if t - timestamp_initial > start_time_range_s:
                index_final = i
                break
        return randrange(index_final)

    def train_load_from_txt_file(self, file_path):
        self.key_stepindex__val_stepdata = {}

        file1 = open(file_path, 'r')
        file_lines = file1.readlines()
        file1.close()

        sizes = []
        prices = []
        timestamps = []
        for line in file_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            line_splitted = line.split(',')
            prices.append(float(line_splitted[0]))
            sizes.append(float(line_splitted[1]))
            timestamps.append(float(line_splitted[3]) / 1000000000.0)

        assert len(prices) == len(sizes) == len(timestamps)
        timestamps.sort()

        prices_observation = []
        sizes_observation = []
        timestamps_observation = []
        # ...........................................................
        step_index = 0
        initial_timestamp_index = self.get_initial_timestamp_index(timestamps)
        timestamp_previous = timestamps[initial_timestamp_index]
        loop_time_noise_range = self.loop_time_fixed * 0.05  # +-5%
        loop_time_tmp = self.loop_time_fixed + random.uniform(-loop_time_noise_range, loop_time_noise_range)
        # ...........................................................
        for p, s, t in zip(prices, sizes, timestamps):
            prices_observation.append(p)
            sizes_observation.append(s)
            timestamps_observation.append(t)
            # loop_time_tmp = self.loop_time_fixed + random.uniform(-loop_time_noise_range, loop_time_noise_range)
            # if t - timestamp_previous >= self.loop_time_fixed and len(prices_observation) > self.observation_seq_len:
            if t - timestamp_previous >= loop_time_tmp and len(prices_observation) > self.observation_seq_len:
                self.key_stepindex__val_stepdata[step_index] = {
                    'prices': prices_observation[-self.observation_seq_len:],
                    'sizes': sizes_observation[-self.observation_seq_len:],
                    'timestamps': timestamps_observation[-self.observation_seq_len:],
                }
                loop_time_tmp = self.loop_time_fixed + random.uniform(-loop_time_noise_range, loop_time_noise_range)
                step_index = step_index + 1
                timestamp_previous = t

                if len(self.key_stepindex__val_stepdata) > self.train_trajectory_len + 300:
                    break

        if len(self.key_stepindex__val_stepdata) > self.train_trajectory_len + 12:
            return True
        return False

    def train_get_observation(self, step_index):
        return self.key_stepindex__val_stepdata[step_index]

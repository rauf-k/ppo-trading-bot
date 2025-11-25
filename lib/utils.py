import os
import random
import json
from datetime import datetime
import numpy as np
from lib import const as CONST


def get_average_position_age(position_durations_list):
    position_durations = []
    for index in range(1, len(position_durations_list)):
        value_previous = position_durations_list[index - 1]
        value_current = position_durations_list[index]
        if value_current < value_previous:
            position_durations.append(value_previous)
    if len(position_durations) == 0:
        return 0.0
    return float(sum(position_durations) / len(position_durations))


def log_handler(epoch_index, symbol, log_data, reports_save_dir=CONST.UTIL_DIR_REPORTS):
    pl_overall_list = []
    position_durations_list = []
    pl_overall = 0.0
    reward_overall = 0.0
    num_trades_overall = 0
    log_lines = []
    for step_index in log_data:
        step_data = log_data[step_index]
        pl_overall = pl_overall + step_data['pl_realized']
        pl_overall_list.append(pl_overall)
        position_durations_list.append(step_data['age_seconds'])
        step_data['pl_overall'] = pl_overall
        reward_overall = reward_overall + step_data['reward']
        step_data['reward_overall'] = reward_overall
        num_trades_overall = num_trades_overall + 1 if step_data['close_signal_received'] else num_trades_overall
        step_data['num_trades_overall'] = num_trades_overall
        keys = list(step_data.keys())
        values = list(step_data.values())
        log_lines.append(','.join(['{},{}'.format(keys[i], values[i]) for i in range(len(keys))]))

    write_lines_to_file(
        log_lines,
        os.path.join(
            reports_save_dir,
            '{}__{}.csv'.format(epoch_index, os.path.basename(symbol).rsplit('.', 1)[0])
        )
    )
    info = 'e,{},sym,{},reward,{},n_trades,{},pl,{},pl_min,{},pl_max,{},trade_time_aver,{}'.format(
        epoch_index,
        symbol,
        round(reward_overall, 3),
        num_trades_overall,
        round(pl_overall, 3),
        round(min(pl_overall_list), 3),
        round(max(pl_overall_list), 3),
        round(get_average_position_age(position_durations_list), 3),
    )
    append_line_to_file(info, os.path.join(reports_save_dir, 'overall_data.csv'))
    print(info.replace(',', '  '))


def append_line_to_file(line, file_path):
    file1 = open(file_path, 'a')
    file1.write('{}\n'.format(line))
    file1.close()


def write_lines_to_file(lines, file_path):
    file1 = open(file_path, 'w')
    file1.writelines(['{}\n'.format(i) for i in lines])
    file1.close()


def read_text_file(file_path):
    file1 = open(file_path, "r")
    file_data = file1.read()
    file1.close()
    return file_data


def get_json_data(file_path):
    file1 = open(file_path)
    json_data = json.load(file1)
    file1.close()
    return json_data


def percent_diff(v_initial, v_final):
    v_initial = 1.0 if v_initial == 0.0 else v_initial
    v_final = 1.0 if v_final == 0.0 else v_final
    pd = (v_final - v_initial) / v_initial
    return pd * 100.0


def data_process_all_diff(market_data, datetime_format='%Y%m%d'):
    trajectory_data = []
    ohlcv = []

    key_stamp__val_date = {
        datetime.strptime(str(date), datetime_format).timestamp(): date for date in list(market_data.keys())}

    timestamps = list(key_stamp__val_date.keys())
    timestamps.sort()
    previous_timestamp = None
    for timestamp in timestamps:
        if previous_timestamp is None:
            previous_timestamp = timestamp
            continue

        data_dict_yesterday = market_data[key_stamp__val_date[previous_timestamp]]
        o_yesterday = float(data_dict_yesterday['o'])
        h_yesterday = float(data_dict_yesterday['h'])
        l_yesterday = float(data_dict_yesterday['l'])
        c_yesterday = float(data_dict_yesterday['c'])
        v_yesterday = float(data_dict_yesterday['v'])

        data_dict_today = market_data[key_stamp__val_date[timestamp]]
        o_today = float(data_dict_today['o'])
        h_today = float(data_dict_today['h'])
        l_today = float(data_dict_today['l'])
        c_today = float(data_dict_today['c'])
        v_today = float(data_dict_today['v'])

        tmp_slice = [
            float(datetime.strptime(str(key_stamp__val_date[timestamp]), datetime_format).date().weekday()) / 4.0,
            np.clip(percent_diff(v_yesterday, v_today) / 200.0, a_min=-1.5, a_max=1.5),
            np.clip(np.log10(o_today + 1.001) / 4.0, a_min=-1.5, a_max=1.5),
            np.clip(np.log10(h_today + 1.001) / 4.0, a_min=-1.5, a_max=1.5),
            np.clip(np.log10(l_today + 1.001) / 4.0, a_min=-1.5, a_max=1.5),
            np.clip(np.log10(c_today + 1.001) / 4.0, a_min=-1.5, a_max=1.5),
            np.clip(np.log10(v_today + 1.001) / 8.0, a_min=-1.5, a_max=1.5),
        ]
        values_yesterday = [o_yesterday, h_yesterday, l_yesterday, c_yesterday]
        values_today = [o_today, h_today, l_today, c_today]
        for value_yesterday in values_yesterday:
            for value_today in values_today:
                value_pd = np.clip(percent_diff(value_yesterday, value_today) / 20.0, a_min=-1.5, a_max=1.5)
                tmp_slice.append(value_pd)

        for index_1, value_today_1 in enumerate(values_today):
            for index_2, value_today_2 in enumerate(values_today):
                if index_1 != index_2:
                    value_pd = np.clip(percent_diff(value_today_1, value_today_2) / 20.0, a_min=-1.5, a_max=1.5)
                    tmp_slice.append(value_pd)

        trajectory_data.append(tmp_slice)
        ohlcv.append([o_today, h_today, l_today, c_today, v_today])
        previous_timestamp = timestamp

    return trajectory_data, ohlcv


def format_observation(symbol, market_data, position_data, reward):
    obs_temp, ohlcv = data_process_all_diff(market_data)
    obs_non_temp = [
        float(position_data['num_days'] / 30.0),
        float(position_data['pl_unrealized'] / 300.0),
        float(position_data['pl_unrealized_max'] / 300.0),
        float(position_data['pl_unrealized_dayli_change'] / 300.0),
        float(position_data['pl_realized'] / 300.0),
        float(reward / 300.0),
        float(np.sign(position_data['position_size_pos_neg'])),
    ]
    return np.array(obs_temp).astype(np.float32), np.array(obs_non_temp).astype(np.float32)


class IntradayObservationProcessor:
    def __init__(self, number_of_points_to_interpolate=32):
        self.number_of_points_to_interpolate = number_of_points_to_interpolate

    def format_observation(self, symbol, market_data, position_data, reward):
        obs_temp = self._format_temporal_observation(symbol, market_data, position_data, reward)
        obs_non_temp = self._format_non_temporal_observation(symbol, market_data, position_data, reward)
        return obs_temp, obs_non_temp

    def _compander(self, x, mu):
        """
        Description : generation encoding mu=255
        """
        # mu = mu-1
        fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        return fx  # np.floor((fx+1)/2*mu+0.5).astype(np.long)

    def _moving_average(self, seq, interval_len):
        return np.convolve(seq, np.ones(interval_len), 'valid') / interval_len

    def _format_temporal_observation(self, symbol, market_data, position_data, reward):
        # x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) ...................................................
        ma_interval_len = 7
        prices = market_data['prices']
        prices_pad = [prices[0] for _ in range(ma_interval_len - 1)] + prices
        prices_sma = self._moving_average(prices_pad, interval_len=ma_interval_len)
        prices_sma_min = np.min(prices_sma)
        prices_sma_max = np.max(prices_sma)
        prices_sma_range = 1.0 if prices_sma_min == prices_sma_max else prices_sma_max - prices_sma_min
        prices_01 = (prices_sma - prices_sma_min) / prices_sma_range
        prices_01_clip = np.clip(prices_01, a_min=0.0, a_max=1.0)
        # ......................................................................................................
        timestamps = market_data['timestamps']
        timestamps_pad = [timestamps[0]] + timestamps
        timestamps_diff = np.diff(timestamps_pad)
        timestamps_diff_clip = np.clip(timestamps_diff, a_min=0.0, a_max=60.0)
        timestamps_diff_clip_log = np.log10(timestamps_diff_clip + 1.0001)
        # ......................................................................................................
        usd = np.array(market_data['prices']) * np.array(market_data['sizes'])
        usd_log = np.log10(usd + 1.0001)
        usd_log_clip = np.clip(usd_log / 12.0, a_min=0.0, a_max=1.0)
        # ......................................................................................................
        obs_temp = np.concatenate((
            np.expand_dims(prices_01_clip, axis=1),
            np.expand_dims(timestamps_diff_clip_log, axis=1),
            np.expand_dims(usd_log_clip, axis=1),
        ), axis=1).astype(np.float32)
        return obs_temp

    def _format_non_temporal_observation(self, symbol, market_data, position_data, reward):
        obs_non_temp = [
            float(position_data['age_seconds'] / 300.0),
            float(position_data['pl_unrealized'] / 30.0),
            float(position_data['pl_realized'] / 30.0),
            float(reward / 30.0),
            float(np.sign(position_data['position_size_pos_neg'])),
        ]
        return np.array(obs_non_temp).astype(np.float32)

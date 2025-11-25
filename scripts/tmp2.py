import os
import numpy as np
import matplotlib.pyplot as plt


def get_file_lines(file_path):
    file1 = open(file_path, 'r')
    file_lines = file1.readlines()
    file1.close()
    return file_lines


def parse_file_lines(file_lines):
    price = []
    size = []
    timestamp_ib = []
    timestamp_local = []
    for line in file_lines:
        line = line.strip()
        if len(line) == 0:
            continue
        line_splitted = line.split(',')
        price.append(float(line_splitted[0]))
        size.append(float(line_splitted[1]))
        timestamp_ib.append(int(line_splitted[2]))
        timestamp_local.append(float(line_splitted[3]) / 1000000000.0)  # in seconds

    # could also sort it here

    return price, size, timestamp_ib, timestamp_local


def sort_to_windows(price, size, ts_ib, ts_local, window_size_in_seconds: float):
    # chop into windows from current time, not from the earliest point
    key_tslocal__val_windowdata = {}
    price_window = []
    size_window = []
    ts_ib_window = []
    ts_local_window = []
    ts_local_previous = ts_local[0]
    for price_val, size_val, ts_ib_val, ts_local_val in zip(price, size, ts_ib, ts_local):
        price_window.append(price_val)
        size_window.append(size_val)
        ts_ib_window.append(ts_ib_val)
        ts_local_window.append(ts_local_val)
        if ts_local_val - ts_local_previous >= window_size_in_seconds:
            key_tslocal__val_windowdata[ts_local_previous] = {
                'price_window': price_window,
                'size_window': size_window,
                'ts_ib_window': ts_ib_window,
                'ts_local_window': ts_local_window,
            }
            price_window = []
            size_window = []
            ts_ib_window = []
            ts_local_window = []
            ts_local_previous = ts_local_val
    return key_tslocal__val_windowdata


def interpolate_single_window(ts_local_window, window_data, number_of_points=10):
    # x = np.array(ts_local_window)
    # y = np.array(window_data)

    xnew = np.linspace(min(ts_local_window), max(ts_local_window), num=number_of_points)
    ynew = np.interp(xnew, ts_local_window, window_data)

    print(window_data)
    print(ynew)
    exit()

    return xnew, ynew


def mean_and_standard_deviation_single_window():

    return 0


def interpolate_all_windows(key_tslocal__val_windowdata):
    key_tslocal__val_window_int = {}
    for ts_local_key in key_tslocal__val_windowdata.keys():
        window_data = key_tslocal__val_windowdata[ts_local_key]
        window_price = window_data['price_window']
        window_size = window_data['size_window']
        # window_ts_ib = window_data['ts_ib_window']
        window_ts_local = window_data['ts_local_window']
        window_price_int = interpolate_single_window(window_ts_local, window_price)
        window_size_int = interpolate_single_window(window_ts_local, window_size)
        window_num_points = float(len(window_ts_local))
        key_tslocal__val_window_int[ts_local_key] = {
            'window_price_int': window_price_int,
            'window_size_int': window_size_int,
            'window_num_points': window_num_points,
        }
    return key_tslocal__val_window_int


date_dir = 'data/train'
for file_name in os.listdir(date_dir):
    file_path = os.path.join(date_dir, file_name)
    file_lines = get_file_lines(file_path)
    price, size, ts_ib, ts_local = parse_file_lines(file_lines)
    key_tslocal__val_windowdata = sort_to_windows(price, size, ts_ib, ts_local, 1.0)

    print(file_name, len(key_tslocal__val_windowdata))

    interpolate_all_windows(key_tslocal__val_windowdata)


    # exit()


# for k in key_tslocal__val_windowdata:
#     print(k, len(key_tslocal__val_windowdata[k]['price_window']))


print('Done!')

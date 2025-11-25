

def slice_to_windows(ts_local_s_, price_, size_, window_size_in_seconds: float):
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
    tsls_previous = tsls[0]
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


ts_local_s_ = [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3]
price_ = [10.3, 20.3, 30.3, 40.3, 50.3, 60.3, 70.3, 80.3]
size_ = [10.7, 20.7, 30.7, 40.7, 50.7, 60.7, 70.7, 80.7]
window_size_in_seconds = 2.0

o = slice_to_windows(ts_local_s_, price_, size_, window_size_in_seconds)
print(o)

exit()
print(key_ts__val_windowdata)
# {3: {'w_l1': [1, 2, 3], 'w_l2': [10, 20, 30]}, 5: {'w_l1': [4, 5], 'w_l2': [40, 50]}, 7: {'w_l1': [6, 7], 'w_l2': [60, 70]}}


exit()
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)




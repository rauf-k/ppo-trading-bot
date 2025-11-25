

seq = [0, 1, 2, 3, 4, 5]
number_of_samples_in_a_batch = 3

observation_window_index = 0

while True:
    timestamps_for_batch = seq[observation_window_index: observation_window_index + number_of_samples_in_a_batch]
    observation_window_index = observation_window_index + 1
    if len(timestamps_for_batch) != number_of_samples_in_a_batch:
        break
    print(timestamps_for_batch)

exit()
for i in range(len(seq) - number_of_samples_in_a_batch + 1):
    print(seq[i: i + number_of_samples_in_a_batch])










exit()
l1_ = [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3]
l2_ = [10.3, 20.3, 30.3, 40.3, 50.3, 60.3, 70.3, 80.3]

seq_index = len(l1_) - 1
print(seq_index, '\n')

most_recent_value1 = l1_[-1]
print(most_recent_value1, '\n')

w_l1 = []
w_l2 = []
key_ts__val_windowdata = {}
while seq_index >= 0:
    value1 = l1_[seq_index]
    value2 = l2_[seq_index]

    w_l1.append(value1)
    w_l2.append(value2)

    if most_recent_value1 - value1 >= 2:
        # print()

        key_ts__val_windowdata[value1] = {
            'w_l1': w_l1,
            'w_l2': w_l2,
        }
        w_l1 = []
        w_l2 = []
        most_recent_value1 = value1

    # print(seq_index, value1, value2)

    seq_index = seq_index - 1

print(key_ts__val_windowdata)






exit()
l2_ = [10.3, 20.3, 30.3, 40.3, 50.3, 60.3, 70.3, 80.3]

l1 = l1_.copy()
l2 = l2_.copy()
l1.reverse()
l2.reverse()
key_ts__val_windowdata = {}
w_l1 = []
w_l2 = []
l1_previous = l1[0]
for val1, val2 in zip(l1, l2):
    w_l1.append(val1)
    w_l2.append(val2)
    if l1_previous - val1 >= 2:
        w_l1.reverse()
        w_l2.reverse()
        key_ts__val_windowdata[val1] = {
            'w_l1': w_l1,
            'w_l2': w_l2,
        }
        w_l1 = []
        w_l2 = []
        l1_previous = val1


keys = list(key_ts__val_windowdata.keys())
keys.sort()

fin = {}

for k in keys:
    # print(key_ts__val_windowdata[k])
    fin[k] = key_ts__val_windowdata[k]

print(fin)

print(l1_)
print(l2_)



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




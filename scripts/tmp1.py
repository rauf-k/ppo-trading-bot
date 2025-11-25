import numpy as np

def average_position_age(position_durations_list):
    position_durations = []
    for index in range(1, len(position_durations_list)):
        value_previous = position_durations_list[index - 1]
        value_current = position_durations_list[index]
        if value_current < value_previous:
            position_durations.append(value_previous)
    if len(position_durations) == 0:
        return 0.0
    return float(sum(position_durations) / len(position_durations))


av = average_position_age([0,1,2,3,2,3,4,5,0,0,0,1,2,3,4,0])

print(av)


exit()


def calculate_reward(self, symbol, epoch_index, market_data, position_data):
    percent_boost = 120
    decay_epochs = 333

    reward_hard = 30  # self._reward_hard(symbol, market_data, position_data)

    if epoch_index < decay_epochs:
        pb = float(percent_boost) - (float(percent_boost) / float(decay_epochs)) * float(epoch_index)
    else:
        pb = 0.0

    reward_final = reward_hard + ((abs(reward_hard) / 100.0) * pb)
    # reward_final = reward_hard

    return reward_final


for epoch_index in range(0, 700):
    r = calculate_reward('self', 'symbol', epoch_index, 'market_data', 'position_data')
    print(epoch_index, r)
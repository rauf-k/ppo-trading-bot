import os
import time
import random
from lib.ppo import PPO
from lib import const as CONST
from lib.utils import log_handler
from lib.market import MarketSimIntraday
from lib.account import AccountSimIntraday


api_ppo = PPO()
api_account = AccountSimIntraday(simulation_mode=True)
api_market = MarketSimIntraday(observation_sequence_len=CONST.OBSERVATION_WINDOW_LEN)


load_weights_epoch = 0
if load_weights_epoch != 0:
    api_ppo.load_weights(load_weights_epoch)


train_file_paths = [os.path.join(CONST.UTIL_DIR_TRAIN, i) for i in os.listdir(CONST.UTIL_DIR_TRAIN)]
epoch_index = load_weights_epoch + 1
while True:
    trajectory_file_path = random.choice(train_file_paths)
    trajectory_symbol = os.path.basename(trajectory_file_path).rsplit('.', 1)[0].upper()
    success = api_market.train_load_from_txt_file(trajectory_file_path)
    if not success:
        continue
    log_data = {}
    loop_time = 0.3
    for step_index in range(CONST.STEPS_PER_EPOCH):

        time_initial = time.time()  # .................................................................................
        market_data = api_market.train_get_observation(step_index)
        position_data = api_account.get_position_data(trajectory_symbol, market_data)
        reward = api_ppo.calculate_reward(trajectory_symbol, epoch_index, market_data, position_data)
        action, value = api_ppo.predict_action(trajectory_symbol, step_index, market_data, position_data, reward)
        api_account.execute_action(trajectory_symbol, action, market_data)

        _time = market_data['timestamps'][-1]
        _price = sum(market_data['prices'][-3:]) / len(market_data['prices'][-3:])
        log_data[step_index] = {**position_data, **{
            'reward': reward,
            'action': action,
            'value': value,
            'time': _time,
            'price': _price,
            'loop_time': loop_time,
        }}

        loop_time = time.time() - time_initial  # .....................................................................

        # print('symbol', trajectory_symbol, 'step_index', step_index, 'loop_time', loop_time)

        if step_index == CONST.STEPS_PER_EPOCH - 1:
            market_data = api_market.train_get_observation(step_index)
            position_data = api_account.get_position_data(trajectory_symbol, market_data)
            reward = api_ppo.calculate_reward(trajectory_symbol, epoch_index, market_data, position_data)
            api_ppo.finish_trajectory(trajectory_symbol, market_data, position_data, reward)

    api_ppo.save_weights_if(epoch_index, save_critic=True)
    api_ppo.train_models()
    log_handler(epoch_index, trajectory_symbol, log_data)
    api_account = AccountSimIntraday(simulation_mode=True)
    epoch_index = epoch_index + 1
    if epoch_index > CONST.EPOCHS:
        break

print('Done!')

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.signal

from lib import const as CONST
from lib.utils import format_observation, IntradayObservationProcessor
from lib.model import get_actor_model, get_critic_model


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    def __init__(
            self,
            steps_per_trajectory,
            observation_shape_temporal,
            observation_shape_non_temporal,
            gamma,
            lambda_
    ):
        self.observation_buffer_temporal = np.zeros(
            (steps_per_trajectory,) + observation_shape_temporal, dtype=np.float32)
        self.observation_buffer_non_temporal = np.zeros(
            (steps_per_trajectory,) + observation_shape_non_temporal, dtype=np.float32)
        self.action_buffer = np.zeros(steps_per_trajectory, dtype=np.int32)
        self.advantage_buffer = np.zeros(steps_per_trajectory, dtype=np.float32)
        self.reward_buffer = np.zeros(steps_per_trajectory, dtype=np.float32)
        self.return_buffer = np.zeros(steps_per_trajectory, dtype=np.float32)
        self.value_buffer = np.zeros(steps_per_trajectory, dtype=np.float32)
        self.logprobability_buffer = np.zeros(steps_per_trajectory, dtype=np.float32)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.pointer = 0
        self.trajectory_start_index = 0

    def store(self, observation_temporal, observation_non_temporal, action, reward, value, logprobability):
        self.observation_buffer_temporal[self.pointer] = observation_temporal
        self.observation_buffer_non_temporal[self.pointer] = observation_non_temporal
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lambda_)
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (np.mean(self.advantage_buffer), np.std(self.advantage_buffer), )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            np.array(self.observation_buffer_temporal).astype(np.float32),
            np.array(self.observation_buffer_non_temporal).astype(np.float32),
            np.array(self.action_buffer).astype(np.float32),
            np.array(self.advantage_buffer).astype(np.float32),
            np.array(self.return_buffer).astype(np.float32),
            np.array(self.logprobability_buffer).astype(np.float32),
        )


class PPO:
    def __init__(self):
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=CONST.LEARNING_RATE_POLICY)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=CONST.LEARNING_RATE_VALUE_FUNCTION)

        self.actor = get_actor_model()
        self.critic = get_critic_model()

        self.buffer = Buffer(
            steps_per_trajectory=CONST.STEPS_PER_EPOCH,
            observation_shape_temporal=(CONST.OBSERVATION_WINDOW_LEN, CONST.TEMPORAL_OBSERVATION_CHANNELS),
            observation_shape_non_temporal=(CONST.NON_TEMPORAL_OBSERVATION_DIM,),
            gamma=CONST.GAMMA,
            lambda_=CONST.LAM
        )

        self.iop = IntradayObservationProcessor()

    def logprobabilities(self, logits, a):
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(tf.one_hot(a, CONST.NUMBER_OF_ACTIONS) * logprobabilities_all, axis=1)
        return logprobability

    def call_actor(self, observation):
        logits = self.actor(observation, training=False)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    def call_critic(self, observation):
        value = self.critic(observation, training=False)
        return value

    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits_train = self.actor(observation_buffer, training=True)
            ratio = tf.exp(self.logprobabilities(logits_train, action_buffer) - logprobability_buffer)
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + CONST.CLIP_RATIO) * advantage_buffer,
                (1 - CONST.CLIP_RATIO) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))

        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer, training=True)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def save_weights_if(self, epoch_index, save_critic=False):
        if epoch_index % CONST.SAVE_WEIGHTS_INTERVAL == 0:
            path_actor = os.path.join(CONST.UTIL_DIR_WEIGHTS, '{}__actor.h5'.format(epoch_index))
            self.actor.save_weights(path_actor)
            if save_critic:
                path_critic = os.path.join(CONST.UTIL_DIR_WEIGHTS, '{}__critic.h5'.format(epoch_index))
                self.critic.save_weights(path_critic)

    def finish_trajectory(self, symbol, market_data, position_data, reward):
        # obs_temp, obs_non_temp = format_observation(symbol, market_data, position_data, reward)
        obs_temp, obs_non_temp = self.iop.format_observation(symbol, market_data, position_data, reward)
        value = self.critic([np.array([obs_temp]), np.array([obs_non_temp])], training=False)
        self.buffer.finish_trajectory(value)

    def predict_action(self, symbol, step_index, market_data, position_data, reward, val_mode=False):
        obs_temp, obs_non_temp = self.iop.format_observation(symbol, market_data, position_data, reward)
        logits, action = self.call_actor([np.array([obs_temp]), np.array([obs_non_temp])])
        value = self.critic([np.array([obs_temp]), np.array([obs_non_temp])], training=False)
        if not val_mode:
            logprobability = self.logprobabilities(logits, action)
            self.buffer.store(obs_temp, obs_non_temp, action, reward, value, logprobability)
        return int(action[0].numpy()), float(value[0].numpy())

    def train_models(self):
        (
            observation_buffer_temporal,
            observation_buffer_non_temporal,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = self.buffer.get()
        for _ in range(CONST.TRAIN_ITERATIONS_POLICY):
            self.train_policy(
                [observation_buffer_temporal, observation_buffer_non_temporal],
                action_buffer,
                logprobability_buffer,
                advantage_buffer
            )
        for _ in range(CONST.TRAIN_ITERATIONS_VALUE_FUNCTION):
            self.train_value_function([observation_buffer_temporal, observation_buffer_non_temporal], return_buffer)

    def load_weights(self, epoch_index):
        self.actor.load_weights('weights/{}__actor.h5'.format(epoch_index))
        self.critic.load_weights('weights/{}__critic.h5'.format(epoch_index))

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def _reward_easy(self, symbol, market_data, position_data):
        pl_realized = position_data['pl_realized']
        pl_unrealized = position_data['pl_unrealized']

        if pl_unrealized < 0.0:
            reward = pl_realized + pl_unrealized
        else:
            # reward = pl_realized + (pl_unrealized / 2.0)
            # reward = pl_realized + pl_unrealized
            reward = pl_realized + (pl_unrealized / 1.5)

        # reward = reward / 100.0

        return reward

    def _reward_hard(self, symbol, market_data, position_data):
        pl_realized = position_data['pl_realized']
        pl_unrealized = position_data['pl_unrealized']

        if pl_unrealized < 0.0:
            reward = pl_realized + pl_unrealized
        else:
            reward = pl_realized

        return reward

    def _reward_time_decay(self, symbol, market_data, position_data):
        pl_realized = position_data['pl_realized']
        pl_unrealized = position_data['pl_unrealized']
        position_age_s = position_data['age_seconds']

        if pl_unrealized < 0.0:
            reward = pl_realized + pl_unrealized
        else:
            reward = pl_realized + (pl_unrealized * (1.0 / (position_age_s + 1.0)))

        return reward

    def calculate_reward__(self, symbol, epoch_index, market_data, position_data):  # warmup
        percent_boost = 30
        # decay_epochs = 333

        # reward_basic = self._reward_hard(symbol, market_data, position_data)
        reward_basic = self._reward_easy(symbol, market_data, position_data)

        # if epoch_index < decay_epochs:
        #     pb = float(percent_boost) - (float(percent_boost) / float(decay_epochs)) * float(epoch_index)
        # else:
        #     pb = 0.0

        pb = float(percent_boost)

        reward_final = reward_basic + ((abs(reward_basic) / 100.0) * pb)

        return reward_final

    def calculate_reward(self, symbol, epoch_index, market_data, position_data):

        if CONST.WARMUP:
            percent_boost = 30
            reward_basic = self._reward_easy(symbol, market_data, position_data)
            pb = float(percent_boost)
            reward_final = reward_basic + ((abs(reward_basic) / 100.0) * pb)
            return reward_final
        else:
            pl_realized = position_data['pl_realized']
            pl_unrealized_change = position_data['pl_unrealized_change']
            reward = pl_realized + pl_unrealized_change
            return reward


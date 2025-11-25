import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lib import const as CONST


def get_model_mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def padded_conv1d(x, number_of_kernels, kernel_size, activation):
    pad_size = kernel_size - 1
    x = tf.pad(x, tf.constant([[0, 0,], [pad_size, 0], [0, 0]]), "CONSTANT")
    x = layers.Conv1D(number_of_kernels, kernel_size, activation=activation, padding='valid')(x)
    return x


def get_model_conv1D(
        input_temporal,
        input_non_temporal,
        output_dimension,
        output_activation=None
):
    num_filters = 21
    activation = 'relu'
    #
    pp = layers.Conv1D(num_filters, 1, activation=activation, padding='same')(input_temporal)  # 12000
    pp = layers.MaxPool1D(pool_size=3)(pp)  # 4000

    pp = layers.Conv1D(num_filters, 1, activation=activation, padding='same')(pp)
    pp = layers.MaxPool1D(pool_size=3)(pp)  # 1333

    pp = layers.Conv1D(num_filters, 1, activation=activation, padding='same')(pp)
    pp = layers.MaxPool1D(pool_size=3)(pp)  # 444

    pp = layers.Conv1D(num_filters, 1, activation=activation, padding='same')(pp)
    pp = layers.MaxPool1D(pool_size=3)(pp)  # 148
    #
    x_t_0 = layers.Conv1D(num_filters, 1, activation=activation, padding='same')(pp)
    x_t_1 = padded_conv1d(x_t_0, number_of_kernels=num_filters, kernel_size=2, activation=activation)
    x_t_2 = padded_conv1d(x_t_1, number_of_kernels=num_filters, kernel_size=4, activation=activation)
    x_t_3 = padded_conv1d(x_t_2, number_of_kernels=num_filters, kernel_size=8, activation=activation)
    x_t_4 = padded_conv1d(x_t_3, number_of_kernels=num_filters, kernel_size=16, activation=activation)
    x_t_5 = padded_conv1d(x_t_4, number_of_kernels=num_filters, kernel_size=32, activation=activation)
    x_t_6 = padded_conv1d(x_t_5, number_of_kernels=num_filters, kernel_size=64, activation=activation)
    x_t_7 = padded_conv1d(x_t_6, number_of_kernels=num_filters, kernel_size=128, activation=activation)

    x_t = layers.Add()([x_t_0, x_t_1, x_t_2, x_t_3, x_t_4, x_t_5, x_t_6, x_t_7])
    x_t = layers.Conv1D(num_filters, 1, activation='relu', padding='same')(x_t)
    x_t = x_t[:, -1:, :]
    x_t = layers.Flatten()(x_t)

    x_t = layers.Dense(units=84, activation='relu')(x_t)
    x_t = layers.Dense(units=84, activation='relu')(x_t)
    x_t = layers.Dense(units=84, activation='relu')(x_t)
    x_t = layers.Dense(units=84, activation='relu')(x_t)

    x_nt = layers.Dense(units=20, activation='relu')(input_non_temporal)
    x_nt = layers.Dense(units=20, activation='relu')(x_nt)
    x_nt = layers.Dense(units=20, activation='relu')(x_nt)
    x_nt = layers.Dense(units=20, activation='relu')(x_nt)

    x_con = layers.Concatenate(axis=1)([x_t, x_nt])

    x_con = layers.Dense(units=104, activation='relu')(x_con)
    x_con = layers.Dense(units=104, activation='relu')(x_con)
    x_con = layers.Dense(units=104, activation='relu')(x_con)
    # x_con = layers.Dense(units=242, activation='relu')(x_con)
    # x_con = layers.Dense(units=242, activation='relu')(x_con)
    x_con = layers.Dense(units=34, activation='relu')(x_con)
    x_con = layers.Dense(units=7, activation='relu')(x_con)

    return layers.Dense(units=output_dimension, activation=output_activation)(x_con)


def get_actor_model():
    input_temporal = keras.Input(
        shape=(CONST.OBSERVATION_WINDOW_LEN, CONST.TEMPORAL_OBSERVATION_CHANNELS), dtype=tf.float32)
    input_non_temporal = keras.Input(shape=(CONST.NON_TEMPORAL_OBSERVATION_DIM,), dtype=tf.float32)

    logits = get_model_conv1D(input_temporal, input_non_temporal, CONST.NUMBER_OF_ACTIONS, None)
    # logits = get_model_transformer_v1(input_temporal, input_non_temporal, CONST.NUMBER_OF_ACTIONS, 7)

    actor = keras.Model(inputs=[input_temporal, input_non_temporal], outputs=logits)
    print('*' * 70)
    actor.summary()
    print('*' * 70)
    return actor


def get_critic_model():
    input_temporal = keras.Input(
        shape=(CONST.OBSERVATION_WINDOW_LEN, CONST.TEMPORAL_OBSERVATION_CHANNELS), dtype=tf.float32)
    input_non_temporal = keras.Input(shape=(CONST.NON_TEMPORAL_OBSERVATION_DIM,), dtype=tf.float32)

    logits = get_model_conv1D(input_temporal, input_non_temporal, 1, None)
    # logits = get_model_transformer_v1(input_temporal, input_non_temporal, 1, 7)

    value = tf.squeeze(logits, axis=1)
    critic = keras.Model(inputs=[input_temporal, input_non_temporal], outputs=value)
    print('*' * 70)
    critic.summary()
    print('*' * 70)
    return critic

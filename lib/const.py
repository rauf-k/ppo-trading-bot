
STEPS_PER_EPOCH = 4000
# TRAJECTORY_LENGTH = 4000
EPOCHS = 3000  # 700

# ................................................................................
# WARMUP = True
WARMUP = False
if WARMUP:
    LEARNING_RATE_POLICY = 0.0003 / 10.0
    LEARNING_RATE_VALUE_FUNCTION = 0.001 / 10.0

    TRAIN_ITERATIONS_POLICY = 30
    TRAIN_ITERATIONS_VALUE_FUNCTION = 30
# ................................................................................
else:
    # LEARNING_RATE_POLICY = 0.0003 / 30.0
    # LEARNING_RATE_VALUE_FUNCTION = 0.001 / 30.0

    # TRAIN_ITERATIONS_POLICY = 12  # 30
    # TRAIN_ITERATIONS_VALUE_FUNCTION = 12  # 30

    LEARNING_RATE_POLICY = 0.0003 / 10.0
    LEARNING_RATE_VALUE_FUNCTION = 0.001 / 10.0

    TRAIN_ITERATIONS_POLICY = 30
    TRAIN_ITERATIONS_VALUE_FUNCTION = 30
# ................................................................................

GAMMA = 0.99
LAM = 0.97
"""
https://github.com/miyamotok0105/unity-ml-agents/blob/master/docs/Training-PPO.md#lambda
Lambda
lambd corresponds to the lambda parameter used when calculating the Generalized Advantage Estimate (GAE). 
This can be thought of as how much the agent relies on its current value estimate when calculating an 
updated value estimate. Low values correspond to relying more on the current value estimate (which can be high bias), 
and high values correspond to relying more on the actual rewards received in the environment 
(which can be high variance). The parameter provides a trade-off between the two, 
and the right value can lead to a more stable training process.

Typical Range: 0.9 - 0.95
"""
CLIP_RATIO = 0.2

RENDER = False

# ................................................

SAVE_WEIGHTS_INTERVAL = 3

OBSERVATION_WINDOW_LEN = 12000
NUMBER_OF_ACTIONS = 2

TEMPORAL_OBSERVATION_CHANNELS = 3
NON_TEMPORAL_OBSERVATION_DIM = 5

POSITION_SIZE_USD = 7000

# .................................................
# utils consts

UTIL_DIR_TRAIN = 'data/train'
UTIL_DIR_VAL = 'data/val'
UTIL_DIR_DEBUG = 'data/debug'
UTIL_DIR_REPORTS = 'reports'
UTIL_DIR_WEIGHTS = 'weights'

UTIL_TWS_HOST_IP = 'xxx.xx.xx.xx'
UTIL_TWS_PORT_REAL_MONEY = 7496
UTIL_TWS_PORT_PAPER_MONEY = 7497
UTIL_TWS_CLIENT_ID = 333

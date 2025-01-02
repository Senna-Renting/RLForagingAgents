import numpy as np
# Welfare functions will be put below
def nash_sw(rewards):
    res = 1
    for r in rewards:
        res *= max(0,r) # No negative numbers allowed
    return res
def mean_sw(rewards):
    return np.mean(rewards)
def zero_sw(rewards):
    return 0
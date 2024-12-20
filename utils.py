import numpy as np


def pad_action(act, act_param):
    action = np.zeros((8,))
    action[0] = act
    if act == 0:
        action[[1]] = act_param
    elif act == 1:
        action[[2]] = act_param
    elif act == 2:
        action[[3]] = act_param
    elif act == 3:
        action[[4]] = act_param
    elif act == 4:
        action[[5]] = act_param
    elif act == 5:
        action[[6]] = act_param
    elif act == 6:
        action[[7]] = act_param
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action

import os
import numpy as np
import pandas as pd
import gym
import torch
import datetime
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from agent import PDQNAgent
from utils import pad_action
from rl_control.UnityCameraEnvironment_exp import UnityCameraControlEnv  # UnityCameraEnvironment_test_3
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
render = True
USE_TRAINED = True
backup = 0
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/test', type=str)
    parser.add_argument('--max_steps', default=128, type=int)
    parser.add_argument('--train_eps', default=400, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--epsilon_start', default=0.95, type=float)
    parser.add_argument('--epsilon_decay', default=5000, type=int)
    parser.add_argument('--epsilon_end', default=0.02, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--param_net_lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--layer_actor', default=[128, 256, 128, 64])
    parser.add_argument('--layer_param', default=[128, 256, 128, 64])

    config = parser.parse_args()
    return config

def train(cfg):
    pass

def evaluation(cfg):
    env = UnityCameraControlEnv(a_p=0, a_r=0, e_thres=0.5, interactive=False).unwrapped
    agent = PDQNAgent(state_space=env.observation_space, action_space=env.action_space,
                      epsilon_start=cfg.epsilon_start, epsilon_decay=cfg.epsilon_decay, epsilon_end=cfg.epsilon_end,
                      batch_size=cfg.batch_size, device=cfg.device, gamma=cfg.gamma,
                      actor_kwargs={"hidden_layers": cfg.layer_actor},
                      param_net_kwargs={"hidden_layers": cfg.layer_param}, seed=cfg.seed,
                      )

    rewards = []
    moving_avg_rewards = []
    eps_steps = []
    confidence_list = []
    miss_target_num = []
    main_directory = 'PDQN/EXP/eval'
    os.makedirs(main_directory, exist_ok=True)
    agent.load('epoch_4060')  # main
    for i_eps in range(1, 1 + cfg.train_eps):
        agent.epsilon = 0
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        if render: env.render()
        episode_reward = 0
        for i_step in range(cfg.max_steps):
            act, act_param, all_action_param = agent.choose_action(state)
            action = pad_action(act, act_param)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            if render: env.render()
            episode_reward += reward
            state = next_state
            if done:
                print("episode:{},episode_reward:{},step:{}".format(i_eps+backup, episode_reward, i_step+1))
                break

        # 创建子文件夹，以"exp{i}"为名称，例如exp1, exp2, ...
        experiment_directory = os.path.join(main_directory, f"exp{i_eps}")
        os.makedirs(experiment_directory, exist_ok=True)
        # 创建一个保存结果的JSON文件
        result = {"reward": episode_reward,
                  "seed": env.envSeed,
                  "step": i_step+1,
                  "miss_target_num": env.miss_target_num,
                  "object_num": env.rtn_object_num,
                  "detect_num": env.object_detect_num,
                  "init_area_ratio": env.init_area_ratio,
                  "object_speed_in_unity": env.object_speed,
                  "confidence_list": env.confidence_list,
                  "action_time":env.action_time_list}
        result_file = os.path.join(experiment_directory, "result.json")
        # 将结果保存为JSON文件
        print(result)
        print("====================")
        with open(result_file, "w") as file:
            json.dump(result, file)

        rewards.append(episode_reward)
        eps_steps.append(i_step)
        confidence_list.append(env.confidence_list)
        miss_target_num.append(env.miss_target_num)



if __name__ == '__main__':
    cfg = get_args()
    if cfg.train:
        train(cfg)
    else:
        print("========Evaluation========")
        evaluation(cfg)
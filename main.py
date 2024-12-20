import os
import numpy as np
import gym
import torch
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
from agent import PDQNAgent
from utils import pad_action
from rl_control.UnityCameraEnvironment_exp import UnityCameraControlEnv

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
render = True
USE_TRAINED = True
backup = 0
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/RC', type=str)
    parser.add_argument('--max_steps', default=128, type=int)
    parser.add_argument('--train_eps', default=4060, type=int)
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
    env = UnityCameraControlEnv(a_p=0, a_r=0, e_thres=0.5, interactive=False).unwrapped
    agent = PDQNAgent(state_space=env.observation_space, action_space=env.action_space,
                      epsilon_start=cfg.epsilon_start, epsilon_decay=cfg.epsilon_decay, epsilon_end=cfg.epsilon_end,
                      batch_size=cfg.batch_size, device=cfg.device, gamma=cfg.gamma,
                      actor_kwargs={"hidden_layers": cfg.layer_actor},
                      param_net_kwargs={"hidden_layers": cfg.layer_param}, seed=cfg.seed,
                      )

    rewards = []
    eps_steps = []
    log_dir = "PDQN/logs/RC/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    global backup
    for i_eps in range(1, 1 + cfg.train_eps):
        global USE_TRAINED
        # The environment occasionally crashes and is temporarily used for training recovery. We will continue to optimize the environment in the future
        if USE_TRAINED:
            backup = 4060
            agent.load('epoch_' + str(backup))
            USE_TRAINED = False
        e = epsilon(i_eps + backup)
        agent.epsilon = e
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
            agent.memory.push(
                state, np.concatenate(([act], all_action_param.data)).ravel(), reward, next_state, done)
            episode_reward += reward
            state = next_state
            if i_step % 32 == 0:
                agent.update()
            if done:
                print("episode:{},episode_reward:{},step:{}".format(i_eps+backup, episode_reward, i_step+1))
                break
        rewards.append(episode_reward)
        eps_steps.append(i_step)
        writer.add_scalar('episode_reward', episode_reward, i_eps+backup)

        if (i_eps+backup) % 10 == 0:
            agent.save('epoch_' + str(i_eps+backup))
    writer.close()


def epsilon(x):
    if x > 300:
        return 0.1
    else:
        return 0.9 - (0.8 / 300) * x

if __name__ == '__main__':
    cfg = get_args()
    if cfg.train:
        train(cfg)
    else:
        print("========Evaluation========")


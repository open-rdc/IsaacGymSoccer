from env import Soccer
import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=2, type=int)
parser.add_argument('--headless', action='store_true')

args = parser.parse_args()
args.headless = False

torch.manual_seed(0)
random.seed(0)

env = Soccer(args)
player_num = 4
while True:
    action = torch.randint(0, 9, (args.num_envs * player_num,), device=args.sim_device)
    env.step(action)
    env.reset()

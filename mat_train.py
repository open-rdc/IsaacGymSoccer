#!/usr/bin/env python
from env import Soccer
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
from config import get_config
from runner.soccer_runner import SoccerRunner as Runner
import argparse
import torch

"""Train script for SMAC."""


def make_train_env(all_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--compute_device_id', default=0, type=int)
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
    parser.add_argument('--num_envs', default=all_args.n_rollout_threads, type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--episode_length', default=all_args.episode_length, type=int)
    parser.add_argument('--n_agent', default=all_args.n_agent, type=int)
    args = parser.parse_args()
    args.headless = True
    #args.episode_length = 1000
    envs = Soccer(args)
    return envs

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env_args = {"scenario": all_args.scenario,
                            "n_agent": all_args.n_agent,
                            "reward": "scoring"}
                env = FootballEnv(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.eval_episodes == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.eval_episodes)])


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper')
    parser.add_argument('--n_agent', type=int, default=3)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument('--self_play_interval', type=int, default=200, help="number of switching episodes for self-play")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("mumu config: ", all_args)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    arg_list = ['--seed', '1', '--env_name', 'soccer', '--algorithm_name', 'mat_dec', '--experiment_name', 'single', '--scenario_name', 'self-play', '--n_agent', '3', '--lr', '5e-4', '--entropy_coef', '0.01', '--max_grad_norm', '0.5', '--n_training_threads', '16', '--n_rollout_threads', '2048', '--num_mini_batch', '1', '--episode_length', '100', '--num_env_steps', '1000000000', '--ppo_epoch', '30', '--clip_param', '0.05', '--use_value_active_masks', '--use_policy_active_masks']
    #arg_list = ['--seed', '1', '--env_name', 'soccer', '--algorithm_name', 'mat_dec', '--experiment_name', 'single', '--scenario_name', 'self-play', '--n_agent', '3', '--lr', '5e-4', '--entropy_coef', '0.01', '--max_grad_norm', '0.5', '--n_training_threads', '16', '--n_rollout_threads', '2048', '--num_mini_batch', '1', '--episode_length', '100', '--num_env_steps', '1000000000', '--ppo_epoch', '30', '--clip_param', '0.05', '--use_value_active_masks', '--use_policy_active_masks', '--model_dir', '/home/mirai/IsaacGymSoccer/results/run21/models/transformer_9764.pt']
    main(arg_list)

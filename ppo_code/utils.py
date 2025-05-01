import torch
import random
import numpy as np
import torch
from torch import nn
import gymnasium as gym
import os
# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import gym_utils

def set_random_seeds(seed):
    '''
    Set the random seeds for deterministic behavior
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

def getEnvs(args):
    env_kwargs = args.env_kwargs
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=1, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{args.run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        max_steps_per_vid=100
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{args.run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=max_steps_per_vid, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=max_steps_per_vid, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs, eval_envs

def save_model(agent,run_name,iteration):
    model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = "tanh",
        output_activation = "identity",
        stabilize_output=False
):
    _str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),}
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    
    activation = _str_to_activation[activation]
    output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(layer_init(nn.Linear(in_size, size)))
        layers.append(activation)
        in_size = size

    if not stabilize_output:
        layers.append(layer_init(nn.Linear(in_size, output_size)))
    else:
        layers.append(layer_init(nn.Linear(in_size, output_size),std=0.01*np.sqrt(2)))
    layers.append(output_activation)
    return nn.Sequential(*layers)


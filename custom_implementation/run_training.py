
import os

import time
from dataclasses import dataclass
from typing import Optional
import tyro


from ppo_trainer import PPO_trainer
@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate_actor: float = 3e-4
    """the learning rate of the optimizer"""
    learning_rate_critic: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel environments"""
    num_steps_per_rollout: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    K_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_epsilon: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_minibatch_size: Optional[int] = None

    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    reward_to_go: bool = True
    n_hl_actor: int =3# num of hidden layers
    hl_size_actor =256 # hidden layer sizes
    activation_actor = 'tanh'
    output_activation_actor= 'identity'
    # Critic architecture
    n_hl_critic: int =3# num of hidden layers
    hl_size_critic =256 # hidden layer sizes
    activation_critic= 'tanh'
    output_activation_critic = 'identity'

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: Optional[str] = None



if __name__ == "__main__":

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print("n_iters= ", args.num_iterations)
    if args.eval_minibatch_size is None:
        args.eval_minibatch_size=args.minibatch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        args.run_name = args.exp_name

    ppo_trainer= PPO_trainer(args)
    ppo_trainer.run_training_loop()
    ppo_trainer.close_envs()
    ppo_trainer.close_logger()



    
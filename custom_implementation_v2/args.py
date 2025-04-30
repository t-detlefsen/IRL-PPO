from typing import Optional
from dataclasses import dataclass
@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
   
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "OpenCabinetDrawer-v1"
    """the id of the environment"""
    num_iterations: int = 100
    """the number of iterations"""
    # total_timesteps: int = 10000000
    # """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    # clip_vloss: bool = False
    # """clipping the vloss is not mentioned in the original paper"""
    ent_coef: float = 0.005
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = True

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")

    num_steps= None
    """the number of steps to run in each environment per policy rollout"""




    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
 
    num_eval_steps= None
    """the number of steps to run in each evaluation environment during evaluation"""
    run_name=None

    max_episode_steps=None



    total_timesteps = None
    
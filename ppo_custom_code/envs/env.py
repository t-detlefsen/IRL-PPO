import os
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
import torch 

def make_envs(args: dict, run_name: str):
    '''
    Create environments for training and evaluation

    Args:
        args (dict): Miscellaneous parameters
        run_name (str): Name of experiment
    Return:
        envs (TODO): Training environments
        eval_envs (TODO): Evaluation environments
    '''
    
    # Setup environment arguments
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # Create environments
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs
    )

    # Flatten action space if is gym.space.Dict
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"runs/{os.path.dirname(args.checkpoint)}/test_videos" # Might need modified if used
        print(f"Saving eval videos to {eval_output_dir}")

        # Are these going to overwrite videos every run?
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30
            )

        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    return envs, eval_envs


if __name__ == "__main__":
    import time
    from dataclasses import dataclass
    import tyroty

    @dataclass
    class Args:
        env_id: str = "OpenCabinetDrawer-v1"
        num_envs: int = 1
        num_eval_envs: int = 2
        evaluate: bool = False
        checkpoint: str = ""
        capture_video: bool = False
        save_train_video_freq: int = 10
        num_steps: int = 50
        num_eval_steps: int = 50
        partial_reset: bool = True
        eval_partial_reset: bool = False
        reconfiguration_freq: int = None
        eval_reconfiguration_freq: int = 1
        control_mode: str = "pd_joint_delta_pos"

    args = tyro.cli(Args) # Is this supposed to be tyroty or is the import wrong?
    run_name = f"{args.env_id}_{int(time.time())}"

    envs, eval_envs = make_envs(args, run_name)

    obs, _ = envs.reset(seed=42)
    print("Env initialized. Obs shape:", obs.shape)

    eval_obs, _ = eval_envs.reset(seed=42)
    print("Eval env initialized. Eval obs shape:", eval_obs.shape)
    # FOR EXAMPLE VIDEO
    for _ in range(60):
        actions = torch.zeros((args.num_envs,) + envs.single_action_space.shape)
        obs, reward, done, trunc, info = envs.step(actions)
    envs.close()
    eval_envs.close()
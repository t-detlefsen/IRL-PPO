from torch.utils.tensorboard import SummaryWriter
from utils import *
from ppo_agent import PPO
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym

def getEnvs(args):
        
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    # if args.control_mode is not None:
    #     env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id,
                    num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq,
                    **env_kwargs)
    eval_envs = gym.make(args.env_id, 
                         num_envs=args.num_eval_envs, 
                         reconfiguration_freq=args.eval_reconfiguration_freq, 
                         **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{args.run_name}/videos"
        print(f"Saving eval videos to {eval_output_dir}")
       
        eval_envs = RecordEpisode(eval_envs,
                                output_dir=eval_output_dir,
                                save_trajectory=args.evaluate,
                                trajectory_name="trajectory",
                                max_steps_per_video=args.num_eval_steps, 
                                video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs, eval_envs

class Logger:
    def __init__(self, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
    def add_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

class PPO_trainer(object):
    '''
    Train an RL agent using PPO
    '''
    def __init__(self, args):
        
        set_random_seeds(args.seed)
        init_gpu()

        self.args=args
        self.envs , self.eval_envs = getEnvs(args)
        self.ppo_agent = PPO(args,
            obs_dim=self.envs.single_observation_space.shape[0],
            act_dim=self.envs.single_action_space.shape[0],
            log_std=-0.5,
            )
        self.num_training_iter= args.num_iterations
        self.num_steps_per_rollout=args.num_steps_per_rollout

        writer = SummaryWriter(f"runs/{args.run_name}")
        self.logger=Logger(writer)

    def collect_training_trajectories(self, agent, batch_size):

        print("\nCollecting data to be used for training...")
        
        training_trajs,envsteps_this_batch= sample_trajectories(env=self.envs,
                                                                agent=agent,
                                                                min_timesteps_per_batch=batch_size,
                                                                max_path_length=self.num_steps_per_rollout,
                                                                seed=self.args.seed
                                                                )
        

        return training_trajs, envsteps_this_batch
    def run_training_loop(self):
        '''
        Train the RL agent usng PPO
        '''
        # TODO: Loop through iterations
            # ----------- TRAINING -----------
            # TODO: Reset train environment
            # TODO: Loop through steps
                # TODO: Generate action
                # TODO: Interact w/ environment
                # TODO: Update agent
            # TODO: Evaluate performance

            # ---------- EVALUATION ----------
            # TODO: Evaluate if specified
                # TODO: Reset eval environment
                # TODO: Loop through steps
                    # TODO: Generate action
                    # TODO: Interact w/ environment
                # TODO: Evaluate performance
                # TODO: Save model?
        total_env_steps=0
        for iteration in range(1, self.num_training_iter + 1):
            print("\n\n********** Iteration %i ************"%iteration)
            paths, env_steps_thisbatch = self.collect_training_trajectories(
                                                                            agent=self.ppo_agent,
                                                                            batch_size=self.args.batch_size
                                                                            )
            
            total_env_steps +=env_steps_thisbatch
            self.ppo_agent.add_to_replay_buffer(paths)
            self.ppo_agent.train_agent_singlebatch()

            #EVALUATION
            with torch.no_grad():
                eval_trajs,_= sample_trajectories(env=self.eval_envs,
                                                                agent=self.ppo_agent,
                                                                min_timesteps_per_batch=self.args.num_eval_steps,
                                                                max_path_length=self.args.num_eval_steps,
                                                                seed=self.args.seed
                                                                )
                eval_rwds=[eval_traj["reward"].sum() for eval_traj in eval_trajs]

                self.logger.add_scalar(tag="Eval_AverageReturn", scalar_value=np.mean(eval_rwds),step=iteration)
                self.logger.add_scalar(tag="Eval_StdReturn", scalar_value=np.std(eval_rwds),step=iteration)

    def close_envs(self):
        self.envs.close()
        self.eval_envs.close()
    def close_logger(self):
        self.logger.close()

                



        

    




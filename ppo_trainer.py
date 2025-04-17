
from utils import *
from ppo_agent import PPO
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import os
import gymnasium as gym
class PPO_trainer(object):
    '''
    Train an RL agent using PPO
    '''
    def __init__(self, args):
        
        set_random_seeds(args.seed)
        init_gpu()

       
        self.envs , self.eval_envs = getEnvs(args)
        self.ppo_agent = PPO(args,
            obs_dim=self.envs.single_observation_space.shape[0],
            act_dim=self.envs.single_action_space.shape[0],
            log_std=-0.5,
            )
        self.num_training_iter= args.num_iterations
        # TODO: Setup environment

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
        for iteration in range(1, self.num_training_iter + 1):
            self.ppo_agent.train_agent_singlebatch()

        pass

def getEnvs(args):
    
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    # if args.control_mode is not None:
    #     env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{args.run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        # if args.save_train_video_freq is not None:
        #     save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
        #     envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs, eval_envs



def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
    """
    :param n_iter:  number of (dagger) iterations
    :param collect_policy:
    :param eval_policy:
    :param initial_expertdata:
    :param relabel_with_expert:  whether to perform dagger
    :param start_relabel_with_expert: iteration at which to start relabel with expert
    :param expert_policy:
    """

    # init vars at beginning of training
    self.total_envsteps = 0
    self.start_time = time.time()

    for itr in range(n_iter):
        print("\n\n********** Iteration %i ************"%itr)

        # decide if videos should be rendered/logged at this iteration
        if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
            self.log_video = True
        else:
            self.log_video = False

        # decide if metrics should be logged
        if self.params['scalar_log_freq'] == -1:
            self.log_metrics = False
        elif itr % self.params['scalar_log_freq'] == 0:
            self.log_metrics = True
        else:
            self.log_metrics = False

        # collect trajectories, to be used for training
        training_returns = self.collect_training_trajectories(itr,
                            initial_expertdata, collect_policy,
                            self.params['batch_size'])
        paths, envsteps_this_batch, train_video_paths = training_returns
        self.total_envsteps += envsteps_this_batch

        # add collected data to replay buffer
        self.agent.add_to_replay_buffer(paths)

        # train agent (using sampled data from replay buffer)
        train_logs = self.train_agent()

        # log/save
        if self.log_video or self.log_metrics:
            # perform logging
            print('\nBeginning logging procedure...')
            self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

            if self.params['save_params']:
                self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))
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




def calculate_ppo_loss():
    return



def collect_training_data(
                            agent,
                            envs,
                            num_steps,
                            num_envs,
                            device,
                            obs_shape,
                            act_shape,
                            global_step,
                            logger,
                            seed):
    next_obs, _ = envs.reset(seed=seed)
   
    next_done = torch.zeros(num_envs, device=device)
    obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + act_shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    final_values = torch.zeros((num_steps, num_envs), device=device)
        
    for step in range(0, num_steps):
        global_step += num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(agent.clip_action(action))
        next_done = torch.logical_or(terminations, truncations).to(torch.float32)
        rewards[step] = reward.view(-1) 

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            with torch.no_grad():
                final_values[step, torch.arange(num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
    return obs, actions, logprobs, rewards,dones, values, final_values, next_obs, next_done, global_step, logger





def compute_advantages(agent,
                        num_steps,
                        gae, 
                        next_obs, 
                        rewards,
                        dones,
                        values,
                        final_values, 
                        next_done, 
                        gae_lambda, 
                        gamma):
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                if gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + gae_lambda * lam_coef_sum
                    reward_term_sum = gae_lambda * gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = gae_lambda * gamma * value_term_sum + gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values
        return advantages, returns



















def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = 'tanh',
        output_activation = 'identity',
):
    _str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),}
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
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

device = torch.device("cuda")
def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(arr):
    return torch.from_numpy(arr).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
def get_device():
    return device

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def _discounted_cumsum(rewards,gamma):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        
        traj_length = len(rewards)
        time_indxs = np.arange(traj_length)
        time_col_vect=time_indxs.reshape(-1,1) #column vector
        diff_matrix= time_indxs-time_col_vect #matrix of differences shape (traj_length, traj_length)
        disc_fact_matrix= np.power(gamma, diff_matrix) #matrix of discount factors shape (traj_length, traj_length)
        disc_fact_matrix=np.triu(disc_fact_matrix)#need the upper triangular matrix vals
     
        rtgs=disc_fact_matrix @ rewards
        return rtgs.flatten() 

def _discounted_return(rewards,gamma):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """

    # TODO: create discounted_returns
    tot_disc_rtn=0
    t_prime=0
    for rwd in rewards:
        disc_rtn_t=rwd*(gamma**t_prime)#disc return at a specific timestep
        tot_disc_rtn+=disc_rtn_t
        t_prime+=1 
    discounted_returns= np.full(len(rewards), tot_disc_rtn) 

    return discounted_returns

def calculate_q_vals( rewards_list,gamma,rtg:bool=True):

    """
        Monte Carlo estimation of the Q function.
    """
    q_vals=[]
    if not rtg:
        #use the whole traj for each timestep
        for rwds_traj in rewards_list:
            discounted_rtns = _discounted_return(rwds_traj,gamma)#returns an array with tot discounted return for each timestep
            #total return is the same for every timestep because we are doing trajectory based
            # Q value is sum of all future rewards in the trajectory 
            q_vals.append(discounted_rtns)

    # Case 2: reward-to-go 
    # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
    else:
        for rwds_traj in rewards_list:
            q_vals_traj=_discounted_cumsum(rwds_traj,gamma)
            q_vals.append(q_vals_traj)
    q_vals= np.concatenate(q_vals) #creates 1D array of all q vals
    return q_vals  # return an array


def sample_trajectory(env, agent, num_steps_per_rollout, seed:int):
    ob, _ = env.reset(seed=seed)
    ob=to_numpy(ob)

    obs, acs, rewards, terminals, vals = [], [], [], [], []
    steps = 0
    while True:

        obs.append(ob)
        ac , val= agent.get_action_and_value(ob)# HINT: query the policy's get_action function [OK]
        vals.append(val)
        acs.append(ac)

        # take that action and record results
        ob, rew, done,_,_ = env.step(ac)
        ob=to_numpy(ob)
        rew=rew.item()
        done=done.item()
       

        # record result of taking that action
        steps += 1
        rewards.append(rew)

        # HINT: rollout can end due to max_path_length
        if (steps>=num_steps_per_rollout) or (done==True):
            rollout_done = 1 
        else:
            rollout_done = 0
        terminals.append(rollout_done)

        if rollout_done:
            break
    return Path(obs, acs,vals, rewards, terminals)

def sample_trajectories(env, agent, min_timesteps_per_batch, max_path_length, seed:int):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path=sample_trajectory(env, agent, max_path_length, seed)
        paths.append(path)
        timesteps_this_batch = timesteps_this_batch+get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch



############################################
############################################

def Path(obs, acs,values, rewards, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    
    return {"observation" : np.array(obs, dtype=np.float32),
            "value": np.array(values, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    values = np.concatenate([path["value"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, values, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
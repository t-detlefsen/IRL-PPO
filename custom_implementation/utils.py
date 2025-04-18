import torch
import random
import numpy as np
import torch
from torch import nn
import time

def set_random_seeds(seed:int = 0):
    '''
    Set the random seeds for deterministic behavior
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True




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


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


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
        # print("disc_factor_matrix= ",disc_fact_matrix.shape)
        # print("rwds shape= ",rewards.shape)
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


def sample_trajectory(env, policy, num_steps_per_rollout, seed:int):
    ob, _ = env.reset(seed=seed)
    ob=to_numpy(ob)
    print(ob)
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # if render:  # feel free to ignore this for now
        #     if 'rgb_array' in render_mode:
        #         if hasattr(env.unwrapped, 'sim'):
        #             if 'track' in env.unwrapped.model.camera_names:
        #                 image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
        #             else:
        #                 image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
        #         else:
        #             image_obs.append(env.render(mode=render_mode))
        #     if 'human' in render_mode:
        #         env.render(mode=render_mode)
        #         time.sleep(env.model.opt.timestep)
        obs.append(ob)
        ac = policy.select_action(ob)# HINT: query the policy's get_action function [OK]
        #print ("multi action size= ",ac_multi.shape)
       
        
        acs.append(ac)

        # take that action and record results
        ob, rew, done,_,_ = env.step(ac)
        print(ob)
        print(rew)
        print(done)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

       
        # HINT: rollout can end due to max_path_length
        if (steps>=num_steps_per_rollout) or (done==True):
            rollout_done = 1 
        else:
            rollout_done = 0
        terminals.append(rollout_done)

        if rollout_done:
            break
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, seed:int):
    #  get this from hw1

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path=sample_trajectory(env, policy, max_path_length, seed)
        paths.append(path)
        timesteps_this_batch = timesteps_this_batch+get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    #  get this from hw1
    sampled_paths = []

    for traj_idx in range(ntraj):
        sampled_paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode))

    return sampled_paths


############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    
    return {"observation" : np.array([o.cpu().numpy() for o in obs], dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array([r.cpu().numpy() for r in rewards], dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array([no.cpu().numpy() for no in next_obs], dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
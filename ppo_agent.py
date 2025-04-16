import torch
from torch import nn


class PPO_Agent:
    '''
    Actor-Critic Method trained via PPO
    '''
    def __init__(self,
                obs_dim:int,
                act_dim:int,
                seed:int = 0,

                #Actor training params
                lr_actor: float = 0.0003,
                
                #critic training params
                lr_critic: float = 0.0003,

                #Overall Training params
                tot_timesteps: int = 10000000,
                n_updates_per_batch: int=4, # same as K_epochs
                gamma: float = 0.8, #discount factor
                clip_epsilon: float =0.2,
                gae_lambda: float = 0.95,
                act_std_init: float = 0.6):
        # TODO: Initialize params
        # TODO: Initialize Replay Buffer
        # TODO: Initialize Old Policy (Actor / Critic)
        # TODO: Initialize Policy (Actor / Critic)
        # TODO: Initialize Optimizer + Loss

        self.gamma=gamma
        self.clip_epsilon=clip_epsilon
        self.loss_function = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

        self.policy = ActorCritic() #ACTOR
        self.old_policy = ActorCritic() #NEED TO CHECK IF I NEED THIS. SAME THING AS POLICY JUST 1 step behind
        self.critic=ActorCritic() # VALUE FUNCTION

    def select_action(self, state: torch.tensor) -> torch.tensor:
        '''
        Select an action using the agent

        Args:
            state (torch.tensor): tensor of current state (TODO: dims)
        Returns:
            action (torch.tensor): tensor of action (TODO: dims)
        '''
        # TODO: Query Old Policy
        # TODO: Add information to replay buffer
        # TODO: Return action

        action = torch.zeros(1)

        return action

    def update(self):
        '''
        Update the agent using PPO
        '''
        # TODO: Loop through buffer rewards backwards
            # TODO: Determine discounted reward + and add to list

        # TODO: Normalize rewards

        # TODO: Calculate Advantages

        # TODO: Loop through epochs
            # TODO: Evaluate policy on old policy states + actions
            # TODO: Calculate ratio
            # TODO: Calculate Surrogate Loss + Clip
            # TODO: Take gradient step

        # TODO: Update old policy
        # TODO: Clear replay buffer

        pass






###################### CODE FROM HW2 ########################
def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log
        q_vals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_vals, terminals)
        train_log = self.actor.update(observations, actions, advantages,q_vals)
        

        return train_log


def calculate_q_vals(self, rewards_list):

    """
        Monte Carlo estimation of the Q function.
    """

    # TODO: return the estimated qvals based on the given rewards, using
        # either the full trajectory-based estimator or the reward-to-go
        # estimator

    # HINT1: rewards_list is a list of lists of rewards. Each inner list
        # is a list of rewards for a single trajectory.
    # HINT2: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these). These
        # functions should only take in a single list for a single trajectory.

    # Case 1: trajectory-based PG
    # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
    # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
    # ordering as observations, actions, etc.
    q_vals=[]
    if not self.reward_to_go:
        #use the whole traj for each timestep
        for rwds_traj in rewards_list:
            discounted_rtns = self._discounted_return(rwds_traj)#returns an array with tot discounted return for each timestep
            #total return is the same for every timestep because we are doing trajectory based
            # Q value is sum of all future rewards in the trajectory 
            
            q_vals.append(discounted_rtns)


    # Case 2: reward-to-go PG
    # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
    else:
        for rwds_traj in rewards_list:
            q_vals_traj=self._discounted_cumsum(rwds_traj)
            q_vals.append(q_vals_traj)
    q_vals= np.concatenate(q_vals) #creates 1D array of all q vals
    return q_vals  # return an array

def estimate_advantage(self, obs, rewards_list, q_values, terminals):

    """
        Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
    """

    # Estimate the advantage when nn_baseline is True,
    # by querying the neural network that you're using to learn the value function
    if self.nn_baseline:

        values_normalized = self.actor.run_baseline_prediction(obs)
        ## ensure that the value predictions and q_values have the same dimensionality
        ## to prevent silent broadcasting errors
        assert values_normalized.ndim == q_values.ndim
        ## TODO: values were trained with standardized q_values, so ensure
            ## that the predictions have the same mean and standard deviation as
            ## the current batch of q_values
        values= unnormalize(values_normalized, q_values.mean(), q_values.std())
        
        
        if self.gae_lambda is not None:
            ## append a dummy T+1 value for simpler recursive calculation
            values = np.append(values, [0])

            ## combine rews_list into a single array
            rewards = np.concatenate(rewards_list)

            ## create empty numpy array to populate with GAE advantage
            ## estimates, with dummy T+1 value for simpler recursive calculation
            batch_size = obs.shape[0] #Equal to T
            advantages = np.zeros(batch_size + 1)

            for i in reversed(range(batch_size)):
                ## TODO: recursively compute advantage estimates starting from
                    ## timestep T.
                ## HINT 1: use terminals to handle edge cases. terminals[i]
                    ## is 1 if the state is the last in its trajectory, and
                    ## 0 otherwise.
                ## HINT 2: self.gae_lambda is the lambda value in the
                    ## GAE formula
                non_terminal_flag=1-terminals[i]#0 when its the last in its trajectory
                TD_error= rewards[i] + self.gamma *non_terminal_flag* values[i+1] - values[i] #temporal difference error / delta in the hw pdf
                advantages[i] = TD_error + self.gamma * self.gae_lambda * non_terminal_flag * advantages[i+1] #recursive calculation of the advantage form pdf. non_terminal_flag ensures advantages arent used at ends of trajectories


            # remove dummy advantage
            advantages = advantages[:-1]

        else:
            ## TODO: compute advantage estimates using q_values, and values as baselines
            # raise NotImplementedError
            advantages = q_values - values

    # Else, just set the advantage to [Q]
    else:
        advantages = q_values.copy()

    # Normalize the resulting advantages
    if self.standardize_advantages:
        ## TODO: standardize the advantages to have a mean of zero
        ## and a standard deviation of one

        
        advantages = normalize(advantages, advantages.mean(), advantages.std())

    return advantages

#####################################################
#####################################################

def add_to_replay_buffer(self, paths):
    self.replay_buffer.add_rollouts(paths)

def sample(self, batch_size):
    return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

#####################################################
################## HELPER FUNCTIONS #################
#####################################################

def _discounted_return(self, rewards):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """

    # TODO: create discounted_returns
    tot_disc_rtn=0
    t_prime=0
    for rwd in rewards:
        disc_rtn_t=rwd*(self.gamma**t_prime)#disc return at a specific timestep
        tot_disc_rtn+=disc_rtn_t
        t_prime+=1 
    discounted_returns= np.full(len(rewards), tot_disc_rtn) 

    return discounted_returns

def _discounted_cumsum(self, rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """

    # TODO: create `discounted_cumsums`
    # HINT: it is possible to write a vectorized solution, but a solution
        # using a for loop is also fine
    
    # discounted_cumsums = []
    # for t_prime in range(traj_length):
    #     disc_rtn_t=0
    #     for t in range(t_prime, traj_length):
    #         disc_rtn_t+= rewards[t]*(self.gamma**(t-t_prime))
    #     discounted_cumsums.append(disc_rtn_t)
    # discounted_cumsums = np.array(discounted_cumsums)

    #from https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    # could also do:
#    traj_length = len(rewards)
    # rtgs = np.zeros_like(rewards)
    # for i in reversed(range(traj_length)):
    #     rtgs[i] = rewards[i] + (self.gamma*rtgs[i+1] if i+1 < traj_length else 0)
    # return rtgs
    traj_length = len(rewards)
    time_indxs = np.arange(traj_length)
    time_col_vect=time_indxs.reshape(-1,1) #column vector
    diff_matrix= time_indxs-time_col_vect #matrix of differences shape (traj_length, traj_length)
    disc_fact_matrix= np.power(self.gamma, diff_matrix) #matrix of discount factors shape (traj_length, traj_length)
    disc_fact_matrix=np.triu(disc_fact_matrix)#need the upper triangular matrix vals
    rtgs=disc_fact_matrix @ rewards.reshape(-1,1)
    return rtgs.flatten() 

#Relevant function from HW2.
def update(self, observations, actions, advantages, q_values=None):# multiple trajectories
    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(actions)
    advantages = ptu.from_numpy(advantages)

    # TODO: update the policy using policy gradient
    # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
    # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method
    # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
    # HINT4: use self.optimizer to optimize the loss. Remember to
        # 'zero_grad' first
        #https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    distros = self.forward(observations) #  action distribution at each state
    log_probs = distros.log_prob(actions) #log_pitheta(ait|sit))    shape [batch_size, action_dim]  --  log prob for each action at each time step 
    
    #From openai link:
#     def compute_loss(obs, act, weights):
#       logp = get_policy(obs).log_prob(act)
#       return -(logp * weights).mean()
    loss= -1 * (log_probs * advantages.detach()).mean() #take negative because we are doing gradient ascent. Do detach() so baselines paramaters are not updated 
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.nn_baseline:
        ## TODO: update the neural network baseline using the q_values as
        ## targets. The q_values should first be normalized to have a mean
        ## of zero and a standard deviation of one.

        ## HINT1: use self.baseline_optimizer to optimize the loss used for
            ## updating the baseline. Remember to 'zero_grad' first
        ## HINT2: You will need to convert the targets into a tensor using
            ## ptu.from_numpy before using it in the loss
        q_values = ptu.from_numpy(q_values)
        q_values = normalize(q_values,q_values.mean(), q_values.std())
        base_pred= self.baseline(observations)
        # print("base_pred shape: ",base_pred.shape)
        # print("q_values shape: ",q_values.shape)
        base_loss= self.baseline_loss(base_pred.squeeze(1), q_values)
        self.baseline_optimizer.zero_grad()
        base_loss.backward()
        self.baseline_optimizer.step()

    train_log = {
        'Training Loss': ptu.to_numpy(loss),
    }
    return train_log
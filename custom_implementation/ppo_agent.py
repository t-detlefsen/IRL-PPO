import torch
from torch import nn
from actor_critic import ActorCritic
from replay_buffer import ReplayBuffer
from utils import *
#from torchviz import make_dot



class PPO:
    '''
    Actor-Critic Method trained via PPO
    '''
    def __init__(self,
                 args,
                obs_dim:int,
                act_dim:int,
                log_std: float = -0.5,
                ):
        # TODO: Initialize params
        # TODO: Initialize Replay Buffer
        # TODO: Initialize Old Policy (Actor / Critic)
        # TODO: Initialize Policy (Actor / Critic)
        # TODO: Initialize Optimizer + Loss

        self.gamma=args.gamma
        self.clip_epsilon=args.clip_epsilon
        self.MSE_loss = nn.MSELoss()
        
        self.K_epochs=args.K_epochs
       
        self.gae_lambda=args.gae_lambda
        
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.lr_actor=args.learning_rate_actor
        self.lr_critic=args.learning_rate_critic
        self.train_batch_size=args.minibatch_size
        self.reward_to_go=args.reward_to_go

        
        self.replay_buffer = ReplayBuffer()
        self.policy = ActorCritic(obs_dim=self.obs_dim,
                                  act_dim=self.act_dim,
                                  logstd=log_std,
                                  hl_size_actor=args.hl_size_actor,
                                  hl_size_critic=args.hl_size_critic,
                                  activation_actor=args.activation_actor,
                                  activation_critic=args.activation_critic,
                                  n_hl_actor=args.n_hl_actor,
                                  n_hl_critic=args. n_hl_critic,
                                    output_activation_actor=args.output_activation_actor,
                                    output_activation_critic=args.output_activation_critic,
                                  ) #curr policy
        
        self.old_policy = ActorCritic(obs_dim=self.obs_dim,
                                  act_dim=self.act_dim,
                                  logstd=log_std,) # 1 step old ploicy
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}])

    def select_action(self, state: np.array) -> np.array:
        '''
        Select an action using the agent

        Args:
            state (torch.tensor): tensor of current state (TODO: dims)
        Returns:
            action (torch.tensor): tensor of action (TODO: dims)
            action_logprob (torch.tensor): tensor of action log probability (TODO: dims)
            value (torch.tensor): tensor of state value (TODO: dims)
        '''
        # TODO: Query Old Policy
        # TODO: Return action
        #print(state)
        with torch.no_grad():
            print("in select action", state)
            state = from_numpy(state)
            action, _, _ = self.old_policy.act(state)
            action =to_numpy(action)
        return action

    

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage ,
        # by querying the neural network that you're using to learn the value function
        obs= from_numpy(obs)
        with torch.no_grad():
            values_normalized = self.policy.critic(obs).squeeze(-1)
        values_normalized = values_normalized.reshape(-1) 
        # print("q_val shape= ", q_values.shape)
        # print("qval_ndim= ",q_values.ndim)
        # print("q_val type= ", type(q_values))
        # print("values_normalized_shape= ", values_normalized.shape)
        # print("values_normalized_ndim= ", values_normalized.ndim)
        # print("values_normalized type= ", type(values_normalized))
        assert values_normalized.ndim == q_values.ndim

        
        values = unnormalize(values_normalized, q_values.mean(), q_values.std()).detach().cpu().numpy()

        
        
        
        if self.gae_lambda is not None:
            ## append a dummy T+1 value for simpler recursive calculation
            values = np.append(values, [0])

            ## combine rews_list into a single array
            rewards_flat = np.concatenate(rewards_list).reshape(-1)
            print(rewards_flat.shape)
            ## create empty numpy array to populate with GAE advantage
            ## estimates, with dummy T+1 value for simpler recursive calculation
            batch_size = obs.shape[0] #Equal to T
            #print("batchsize= " , batch_size)
            advantages = np.zeros_like(q_values)

            for i in reversed(range(batch_size)):

                non_terminal_flag=1-terminals[i]#0 when its the last in its trajectory
                TD_error= rewards_flat[i] + self.gamma *non_terminal_flag* values[i+1] - values[i] #temporal difference error / delta in the hw pdf
                advantages[i] = TD_error + self.gamma * self.gae_lambda * non_terminal_flag * advantages[i+1] #recursive calculation of the advantage form pdf. non_terminal_flag ensures advantages arent used at ends of trajectories
            #advantages = advantages[:-1]

        else:
        
            advantages = q_values - values

        advantages = normalize(advantages, advantages.mean(), advantages.std())

        return advantages


    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample_replay_buffer(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
    
    def train_agent_singlebatch(self):

        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
       # Collects batch of data from replay buffer
        ob_batch, ac_batch, re_batch, _, terminal_batch = self.sample_replay_buffer(self.train_batch_size)
       
        q_vals= calculate_q_vals(rewards_list=re_batch,
                                    gamma=self.gamma,
                                    rtg=self.reward_to_go)
        advantages = self.estimate_advantage(obs=ob_batch,
                                                rewards_list=re_batch,
                                                q_values=q_vals,
                                                terminals=terminal_batch)
        T, N, obs_dim  = ob_batch.shape      # (50, 112, obs_dim)  for example
        _, _, act_dim  = ac_batch.shape
        ob_batch       = ob_batch.reshape(-1, obs_dim)
        ac_batch       = ac_batch.reshape(-1, act_dim)
        advantages     = advantages.reshape(-1)
        q_vals         = q_vals.reshape(-1)
        terminal_batch = terminal_batch.reshape(-1)


        ob_batch = from_numpy(ob_batch)
        ac_batch = from_numpy(ac_batch)
        
        re_batch = np.array(re_batch)
        re_batch = re_batch.reshape(-1)
        re_batch=from_numpy(re_batch)
        re_batch=torch.squeeze(re_batch)
        re_batch=re_batch.detach()
        #print("re_batch shape = ", re_batch.shape)
        advantages = from_numpy(advantages)
        
        all_logs = []
        for train_step in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(observation=ob_batch, 
                                                                        action=ac_batch
                                                                                                )

            with torch.no_grad():
                logprobs_old, _, _ = self.old_policy.evaluate(observation=ob_batch, action=ac_batch)
          

            
            ratios = torch.exp(logprobs - logprobs_old)
            #print("state_ values shape= ", state_values.shape)
            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages

            # final loss of clipped objective PPO
            policy_loss=-torch.min(surr1, surr2)
            value_loss=self.MSE_loss(state_values.squeeze(-1), re_batch)
            entropy_loss=- 0.01 * dist_entropy
            
            loss = (policy_loss+ 0.5 *value_loss +entropy_loss).mean()
            
         
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.replay_buffer.clear()
       


            
            

# def update(self):
    #     '''
    #     Update the agent using PPO
    #     '''
    #     # TODO: Loop through buffer rewards backwards
    #         # TODO: Determine discounted reward + and add to list

    #     # TODO: Normalize rewards

    #     # TODO: Calculate Advantages

    #     # TODO: Loop through epochs
    #         # TODO: Evaluate policy on old policy states + actions
    #         # TODO: Calculate ratio
    #         # TODO: Calculate Surrogate Loss + Clip
    #         # TODO: Take gradient step

    #     # TODO: Update old policy
    #     # TODO: Clear replay buffer

    #     pass










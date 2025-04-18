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

    def get_action_and_value(self, state: np.array) -> np.array:
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
            action, _, value = self.old_policy.act(state)
        return action,value

    

    def estimate_advantage(self, obs,val, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage ,
        # by querying the neural network that you're using to learn the value function
      
        values_normalized = val
        print("val shape: ",val.shape)
        print("q_val shape= ",q_values.shape)
        assert values_normalized.ndim == q_values.ndim

        values = unnormalize(values_normalized, q_values.mean(), q_values.std())

        if self.gae_lambda is not None:
            ## append a dummy T+1 value for simpler recursive calculation
            values = np.append(values, [0])

            ## combine rews_list into a single array
            rewards_flat = np.concatenate(rewards_list)
            ## create empty numpy array to populate with GAE advantage
            ## estimates, with dummy T+1 value for simpler recursive calculation
            batch_size = obs.shape[0] #Equal to T
            
            advantages = np.zeros_like(values)

            for i in reversed(range(batch_size)):

                non_terminal_flag=1-terminals[i]#0 when its the last in its trajectory
                TD_error= rewards_flat[i] + self.gamma *non_terminal_flag* values[i+1] - values[i] #temporal difference error / delta in the hw pdf
                advantages[i] = TD_error + self.gamma * self.gae_lambda * non_terminal_flag * advantages[i+1] #recursive calculation of the advantage form pdf. non_terminal_flag ensures advantages arent used at ends of trajectories
            advantages = advantages[:-1]

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

       # Collects batch of data from replay buffer
        observations, actions,vals, rwds, terminals = self.sample_replay_buffer(self.train_batch_size)
       
        q_vals= calculate_q_vals(rewards_list=rwds,
                                    gamma=self.gamma,
                                    rtg=self.reward_to_go)
        advantages = self.estimate_advantage(obs=observations,
                                             val=vals,
                                            rewards_list=rwds,
                                            q_values=q_vals,
                                            terminals=terminals)
        
        advantages=from_numpy(advantages).detach()
        rwds=from_numpy(rwds[0]).detach()
      
        
        for train_step in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(observation=observations, 
                                                                        action=actions)

            with torch.no_grad():
                logprobs_old, _, _ = self.old_policy.evaluate(observation=observations, action=actions)

            ratios = torch.exp(logprobs - logprobs_old)
      
            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages

            # final loss of clipped objective PPO
           
            
            loss = (-torch.min(surr1, surr2)+ 0.5 *self.MSE_loss(state_values.squeeze(), rwds) - 0.01 * dist_entropy).mean()


           
            # print("\n[DEBUG] grad_fn check before backward:")
            # print("logprobs grad_fn:", logprobs.grad_fn)
            # print("logprobs_old grad_fn:", logprobs_old.grad_fn)
            # print("state_values grad_fn:", state_values.grad_fn)
            # print("dist_entropy grad_fn:", dist_entropy.grad_fn)
            # print("advantages grad_fn:", advantages.grad_fn if hasattr(advantages, "grad_fn") else "N/A (likely detached)")
            # print("rwds grad_fn:", rwds.grad_fn if hasattr(rwds, "grad_fn") else "N/A (likely detached)")
            # print("policy_loss grad_fn:", policy_loss.grad_fn)
            # print("value_loss grad_fn:", value_loss.grad_fn)
            # print("entropy_loss grad_fn:", entropy_loss.grad_fn)
            # print("loss grad_fn:", loss.grad_fn)
            # print("dist_entropy:", dist_entropy.grad_fn)


         
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










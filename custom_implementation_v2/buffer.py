import torch
import numpy as np

class Rollout_Data:
    def __init__(self,observations, actions, logprobs, rewards,dones, values, vals_at_eps_end):
        self.observations = observations 
        self.actions = actions
        self.logprobs = logprobs
        self.rewards = rewards
        self.dones = dones
        self.values = values
        self.vals_at_eps_end =vals_at_eps_end

        self.advantages=None
        self.returns=None

    
    def compute_advantages_and_returns(self, num_steps, gamma, use_gae,use_fh_gae, gae_lambda, device="cuda"):
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(device)
          
            if use_gae:
                prev_adv = 0
                for t in reversed(range(num_steps)):
                    not_done = 1.0 - self.dones[t + 1]
                    curr_val = not_done * self.values[t+1] + self.vals_at_eps_end[t+1]
                    prev_value= self.values[t]
                    rwd= self.rewards[t]

                    td_error = rwd + (gamma * curr_val) - prev_value
                    adv = td_error + (gamma * gae_lambda * not_done * prev_adv)
                    advantages[t] = adv
                    prev_adv=adv

            elif use_fh_gae:
                lambda_geometric_sum = 0.
                rwd_weightedsum = 0. # λ-weighted average of discounted future rewards
                value_weightedsum = 0. 
                for t in reversed(range(num_steps)):
                    not_done = 1.0 - self.dones[t + 1]
                    curr_val = not_done * self.values[t+1] + self.vals_at_eps_end[t+1]
                    prev_value= self.values[t]
                    rwd= self.rewards[t]

                    lambda_geometric_sum = lambda_geometric_sum * not_done # will reset sums if episode terminates
                    rwd_weightedsum = rwd_weightedsum * not_done
                    value_weightedsum = value_weightedsum * not_done

                    lambda_geometric_sum = 1 + (gae_lambda * lambda_geometric_sum) # 1+λ(1+λ+λ^2+⋯) = 1+λ+λ^2+⋯
                    rwd_weightedsum =  (lambda_geometric_sum * rwd) + (gae_lambda * gamma * rwd_weightedsum) # rwd_weightedsum_t = sum_{n=0}^{T - t - 1} lambda^n * (sum_{k=0}^{n - 1} gamma^k * r_{t + k})

                    value_weightedsum =  (gamma * curr_val) + (gae_lambda * gamma * value_weightedsum) # value_weightedsum_t = sum_{n=0}^{T - t - 1} lambda^n * (gamma^n * V(s_{t + n}))

                    advantages[t] = ((rwd_weightedsum + value_weightedsum) / lambda_geometric_sum) - prev_value
           
            returns = advantages + self.values[:-1] #delete the final value from the values list
            self.returns=returns
            self.advantages=advantages
    

    def flatten(self, obs_shape, act_shape):
        self.observations = self.observations.reshape((-1,) + obs_shape)
        self.logprobs = self.logprobs.reshape(-1)
        self.actions = self.actions.reshape((-1,) + act_shape)
        self.advantages = self.advantages.reshape(-1)
        self.returns = self.returns.reshape(-1)
        self.values = self.values.reshape(-1)

 



    

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
        self.num_mb=None
        self.mb_ID= None
    
    def compute_advantages_and_returns(self, num_steps, gamma, use_gae,use_fh_gae, gae_lambda, device="cuda"):
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(device)
            prev_adv = 0
            lam_coef_sum = 0.
            reward_term_sum = 0. 
            value_term_sum = 0. 
            for t in reversed(range(num_steps)):
                not_done = 1.0 - self.dones[t + 1]
                curr_val = not_done * self.values[t+1] + self.vals_at_eps_end[t+1] # for environments where episode ended, will use vals_at_eps_end
                prev_value= self.values[t]
                rwd= self.rewards[t]
                if use_gae:
                    td_error = rwd + (gamma * curr_val) - prev_value
                    adv = td_error + (gamma * gae_lambda * not_done * prev_adv)
                    advantages[t] = adv
                    prev_adv=adv

                elif use_fh_gae:
                    lam_coef_sum = lam_coef_sum * not_done
                    reward_term_sum = reward_term_sum * not_done
                    value_term_sum = value_term_sum * not_done

                    lam_coef_sum = 1 + (gae_lambda * lam_coef_sum)
                    reward_term_sum = (gae_lambda * gamma * reward_term_sum) + (lam_coef_sum * rwd)
                    value_term_sum = (gae_lambda * gamma * value_term_sum) + (gamma * curr_val)

                    advantages[t] = ((reward_term_sum + value_term_sum) / lam_coef_sum) - prev_value
           
            returns = advantages + self.values
            self.returns=returns
            self.advantages=advantages
    

    def flatten(self, obs_shape, act_shape):
        self.observations = self.observations.reshape((-1,) + obs_shape)
        self.logprobs = self.logprobs.reshape(-1)
        self.actions = self.actions.reshape((-1,) + act_shape)
        self.advantages = self.advantages.reshape(-1)
        self.returns = self.returns.reshape(-1)
        self.values = self.values.reshape(-1)

 



    

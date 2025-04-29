import torch
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
    
    def compute_advantages_and_returns(self, num_steps, gamma, use_gae, gae_lambda, device="cuda"):
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                next_not_done = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
                real_next_values = next_not_done * nextvalues + self.vals_at_eps_end[t+1] 
                if use_gae:
                
                    if t == self.num_steps - 1: 
                        lam_coef_sum = 0.
                        reward_term_sum = 0. 
                        value_term_sum = 0. 
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + gae_lambda * lam_coef_sum
                    reward_term_sum = gae_lambda * gamma * reward_term_sum + lam_coef_sum * self.rewards[t]
                    value_term_sum = gae_lambda * gamma * value_term_sum + gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - self.values[t]
                else:
                    delta = self.rewards[t] + gamma * real_next_values - self.values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + self.values
            self.returns=returns
            self.advantages=advantages
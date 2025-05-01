import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from utils import build_mlp

class Agent(nn.Module):
    def __init__(self,args,obs_dim, act_dim):
        super().__init__()
                
        self.actor = build_mlp(
                                input_size=obs_dim,
                                output_size=act_dim,
                                n_layers=args.n_hl_actor,
                                size=args.hl_size_actor,
                                activation=args.activation_actor,
                                output_activation=args.output_activation_actor,
                                stabilize_output=True
                                )
        self.critic = build_mlp(
                                input_size=obs_dim,
                                output_size=1,
                                n_layers=args.n_hl_critic,
                                size=args.hl_size_critic,
                                activation=args.activation_critic,
                                output_activation=args.output_activation_critic
                                )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
 
    def get_value(self, observation):
        return self.critic(observation)
    
    # Used during training rollouts
    def get_action_and_logprob(self, observation):
        action_mean = self.actor(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_distribution = Normal(action_mean, action_std) 
        action_to_take = action_distribution.sample() # during training, sample from a normal distribution to encourage exploration and gain new experiences
        logprob = action_distribution.log_prob(action_to_take).sum(1)
        return action_to_take, logprob
    
    # used in the update loop
    # accepts the actions taken by the old policy
    # recomputes the logprobs using the current policy --> compares past actions with what the current policy would do
    # also returns the entropy for the entropy bonus term
    def get_newlogprob_and_entropy(self, observation, old_action):
        new_action_mean = self.actor(observation)
        new_action_logstd = self.actor_logstd.expand_as(new_action_mean)
        new_action_std = torch.exp(new_action_logstd)
        new_action_distribution = Normal(new_action_mean, new_action_std) 

        entropy=new_action_distribution.entropy().sum(1)
        new_logprob = new_action_distribution.log_prob(old_action).sum(1)
        return new_logprob , entropy
    
    def get_action_for_inference(self, observation):
        action_to_take = self.actor(observation) # during evaluation rollouts, simply take the action outputted by the actor network
        return action_to_take
    



    

    
    
    